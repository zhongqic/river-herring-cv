#!/usr/bin/env python3
"""
Command-line fish counting script using YOLO
Removes live view and adds video saving functionality
"""

from ultralytics import YOLO
import cv2
import time
import os
import numpy as np
import pandas as pd
import supervision as sv
from pathlib import Path
from collections import defaultdict
from datetime import timedelta, datetime
import torch
import argparse


def ensure_dirs(out_dir: Path):
    """Create output directories for RH and Non_RH detections"""
    rh_dir = out_dir / "RH"
    non_rh_dir = out_dir / "Non_RH"
    rh_dir.mkdir(parents=True, exist_ok=True)
    non_rh_dir.mkdir(parents=True, exist_ok=True)
    return rh_dir, non_rh_dir


def init_line_counter(frame_width: int, frame_height: int, line_pos=0.5,
                      move_right: str = "Up", move_left: str = "Down"):
    """Initialize line counter for fish crossing detection"""
    # define the start and end of the line
    start = sv.Point(int(frame_width * line_pos), -250)
    end   = sv.Point(int(frame_width * line_pos), frame_height)
    line_counter = sv.LineZone(start=start, end=end, triggering_anchors=[sv.Position.CENTER])
    line_annot   = sv.LineZoneAnnotator(
        thickness=1, text_thickness=1, text_scale=0.6,
        custom_in_text = move_right, custom_out_text = move_left,
        display_in_count = True, display_out_count = True
    )
    box_annot = sv.BoxAnnotator(thickness=1)
    label_annot = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT,
                                           text_padding = 0)

    return line_counter, line_annot, box_annot, label_annot


def process_video(
    video_path: str,
    weights: str,
    out_dir: str,
    class_id: int = 0,
    conf_thresh: float = 0.7,
    line_pos: float = 0.5,
    imgsz=(480, 320),
    tracker_cfg="bytetrack.yaml",
    move_right: str = "Up",
    move_left: str = "Down",
    save_video: bool = False
):
    """
    Process video for fish counting with optional video output
    
    Args:
        video_path: Input video file path
        weights: YOLO model weights path
        out_dir: Output directory for results
        class_id: Class ID to count (default: 0)
        conf_thresh: Confidence threshold for detections
        line_pos: Position of counting line (0.0 to 1.0)
        imgsz: Input image size for model
        tracker_cfg: Tracker configuration file
        move_right: Label for right movement
        move_left: Label for left movement
        save_video: Whether to save annotated video
    """
    
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rh_dir, non_rh_dir = ensure_dirs(out_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = YOLO(weights).to(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # Get video properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_w}x{frame_h} @ {fps}fps, {total_frames} frames")
    
    # Initialize line counter and annotators
    line_counter, line_annot, box_annot, label_annot = init_line_counter(frame_w, frame_h, line_pos, move_right, move_left)

    # Initialize video writer if saving
    video_writer = None
    if save_video:
        output_video_path = out_dir / f"{video_path.stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (frame_w, frame_h)
        )
        print(f"Saving annotated video to: {output_video_path}")

    # Data collection
    conf_hist, time_hist, pos_hist, size_hist = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    events = []
    count_csv_path = out_dir / f"{video_path.stem}_count.csv"

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting processing: {video_path.name}")

    frame_idx = 0
    start_time = time.time()
    last_progress_time = start_time

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Progress reporting
        current_time = time.time()
        if current_time - last_progress_time >= 5.0 or frame_idx % 100 == 0:  # Every 5 seconds or 100 frames
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            elapsed = current_time - start_time
            fps_processing = frame_idx / elapsed if elapsed > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}) - Processing FPS: {fps_processing:.1f}")
            last_progress_time = current_time

        # YOLO inference + tracking
        results = model.track(frame, persist=True, tracker=tracker_cfg, imgsz=list(imgsz), verbose=False)

        r0 = results[0]
        frame_annot = frame.copy()
        
        # Process detections if any exist
        if r0.boxes is not None and r0.boxes.xywh is not None and r0.boxes.id is not None:
            det = sv.Detections.from_ultralytics(r0)
            det.track_id = r0.boxes.id.cpu().numpy().astype(int)

            # Filter for counting (confidence and class)
            keep = (det.class_id == class_id) & (det.confidence >= conf_thresh)
            det_f = det[keep] if keep.any() else det[:0]

            # Annotate frame with boxes
            frame_annot = box_annot.annotate(scene=frame_annot, detections=det)
            # Annotate frame with labels
            labels = [f"{tracker_id} {class_id} {confidence:0.2f}"
                          for _, _, confidence, class_id, tracker_id, _ in det]
            frame_annot = label_annot.annotate(scene=frame_annot, detections=det, labels = labels)

            # Check line crossings
            crossed_in, crossed_out = line_counter.trigger(detections=det_f)

            # Store detection histories
            boxes_xywh = r0.boxes.xywh.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else np.zeros(len(boxes_xywh))
            clss = r0.boxes.cls.cpu().numpy().astype(int).tolist()
            tids = r0.boxes.id.int().cpu().tolist()
            t_now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            for i, tid in enumerate(tids):
                x, y, w, h = boxes_xywh[i]
                conf_hist[tid].append(float(confs[i]))
                time_hist[tid].append(float(t_now))
                pos_hist[tid].append((float(x), float(y)))
                size_hist[tid].append((float(w), float(h)))

            # Record crossing events
            def add_events(mask, direction):
                idxs = np.where(mask)[0]
                for j in idxs:
                    tid = int(det_f.tracker_id[j])
                    conf = float(det_f.confidence[j])
                    cls = int(det_f.class_id[j])
                    events.append({
                        "frame": frame_idx,
                        "time": str(timedelta(seconds=t_now)),
                        "track_id": tid,
                        "species": model.names[cls],
                        "direction": direction,
                        "confidence": conf
                    })
                    # Save detection image
                    out_subdir = rh_dir if cls == class_id else non_rh_dir
                    cv2.imwrite(str(out_subdir / f"{tid}_{cls}_{direction}_{frame_idx}.png"), frame_annot)

            add_events(crossed_in, move_right)
            add_events(crossed_out, move_left)

        # Annotate with line and counter
        frame_annot = line_annot.annotate(frame=frame_annot, line_counter=line_counter)

        # Write frame to output video if saving
        if video_writer is not None:
            video_writer.write(frame_annot)

    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()

    # Save results
    df = pd.DataFrame(events)
    df.to_csv(count_csv_path, index=False)
    
    # Print summary
    total_time = time.time() - start_time
    total_counts = line_counter.in_count + line_counter.out_count
    
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Processing completed!")
    print(f"Video: {video_path.name}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Processing time: {timedelta(seconds=int(total_time))}")
    print(f"Average processing FPS: {frame_idx/total_time:.1f}")
    print(f"Fish counts - {move_right}: {line_counter.in_count}, {move_left}: {line_counter.out_count}")
    print(f"Total crossings detected: {total_counts}")
    print(f"Results saved to: {count_csv_path}")
    if save_video:
        print(f"Annotated video saved to: {out_dir / f'{video_path.stem}_annotated.mp4'}")


def main():
    parser = argparse.ArgumentParser(description='Fish counting with YOLO - Command Line Interface')
    
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('weights', help='Path to YOLO model weights')
    parser.add_argument('out_dir', help='Output directory for results')
    
    parser.add_argument('--class_id', type=int, default=0, help='Class ID to count (default: 0)')
    parser.add_argument('--conf_thresh', type=float, default=0.7, help='Confidence threshold (default: 0.7)')
    parser.add_argument('--line_pos', type=float, default=0.5, help='Line position 0.0-1.0 (default: 0.5)')
    parser.add_argument('--imgsz', nargs=2, type=int, default=[480, 320], help='Input image size (default: 480 320)')
    parser.add_argument('--tracker', default='bytetrack.yaml', help='Tracker config file (default: bytetrack.yaml)')
    parser.add_argument('--move_right', default='Up', help='Label for rightward movement (default: Up)')
    parser.add_argument('--move_left', default='Down', help='Label for leftward movement (default: Down)')
    parser.add_argument('--save_video', action='store_true', help='Save annotated video')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    if not Path(args.weights).exists():
        print(f"Error: Model weights not found: {args.weights}")
        return 1
    
    try:
        process_video(
            video_path=args.video_path,
            weights=args.weights,
            out_dir=args.out_dir,
            class_id=args.class_id,
            conf_thresh=args.conf_thresh,
            line_pos=args.line_pos,
            imgsz=tuple(args.imgsz),
            tracker_cfg=args.tracker,
            move_right=args.move_right,
            move_left=args.move_left,
            save_video=args.save_video
        )
        return 0
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
