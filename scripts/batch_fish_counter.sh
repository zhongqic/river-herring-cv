#!/bin/bash

# Batch Fish Counter Script
# Processes multiple video files listed in a text file using the fish counting Python script
# Each video's results are saved in separate folders named after the video file

set -e  # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] VIDEO_LIST_FILE WEIGHTS OUT_DIR"
    echo ""
    echo "Process multiple videos for fish counting using YOLO"
    echo ""
    echo "Arguments:"
    echo "  VIDEO_LIST_FILE    Text file containing video file paths (one per line)"
    echo "  WEIGHTS           Path to YOLO model weights"
    echo "  OUT_DIR           Base output directory for all results"
    echo ""
    echo "Options:"
    echo "  --python-script PATH   Path to process_video_cli.py (default: ./process_video_cli.py)"
    echo "  --class-id ID         Class ID to count (default: 0)"
    echo "  --conf-thresh THRESH  Confidence threshold (default: 0.7)"
    echo "  --line-pos POS        Line position 0.0-1.0 (default: 0.5)"
    echo "  --imgsz W H          Input image size (default: 480 320)"
    echo "  --tracker CONFIG     Tracker config file (default: bytetrack.yaml)"
    echo "  --move-right LABEL   Label for rightward movement (default: Up)"
    echo "  --move-left LABEL    Label for leftward movement (default: Down)"
    echo "  --save-video         Save annotated videos"
    echo "  --continue-on-error  Continue processing even if a video fails"
    echo "  --dry-run           Show what would be processed without actually running"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 video_list.txt model.pt /output/results --save-video"
    echo ""
    echo "Video list file format (one video path per line):"
    echo "  /path/to/video1.mp4"
    echo "  /path/to/video2.avi"
    echo "  /path/to/video3.mov"
}

# Default values
PYTHON_SCRIPT="./src/count_fish.py"
CLASS_ID=0
CONF_THRESH=0.7
LINE_POS=0.5
IMGSZ_W=480
IMGSZ_H=320
TRACKER="bytetrack.yaml"
MOVE_RIGHT="Up"
MOVE_LEFT="Down"
SAVE_VIDEO=false
CONTINUE_ON_ERROR=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python-script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        --class-id)
            CLASS_ID="$2"
            shift 2
            ;;
        --conf-thresh)
            CONF_THRESH="$2"
            shift 2
            ;;
        --line-pos)
            LINE_POS="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ_W="$2"
            IMGSZ_H="$3"
            shift 3
            ;;
        --tracker)
            TRACKER="$2"
            shift 2
            ;;
        --move-right)
            MOVE_RIGHT="$2"
            shift 2
            ;;
        --move-left)
            MOVE_LEFT="$2"
            shift 2
            ;;
        --save-video)
            SAVE_VIDEO=true
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Check required arguments
if [[ $# -lt 3 ]]; then
    echo "Error: Missing required arguments" >&2
    usage
    exit 1
fi

VIDEO_LIST_FILE="$1"
WEIGHTS="$2"
OUT_DIR="$3"

# Validate inputs
if [[ ! -f "$VIDEO_LIST_FILE" ]]; then
    echo "Error: Video list file not found: $VIDEO_LIST_FILE" >&2
    exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
    echo "Error: Model weights file not found: $WEIGHTS" >&2
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT" >&2
    echo "Make sure process_video_cli.py is in the current directory or specify path with --python-script" >&2
    exit 1
fi

# Create base output directory
mkdir -p "$OUT_DIR"

# Function to get filename without extension
get_basename() {
    local filepath="$1"
    local filename=$(basename "$filepath")
    echo "${filename%.*}"
}

# Function to process a single video
process_video() {
    local video_path="$1"
    local video_basename="$2"
    local video_out_dir="$3"
    
    echo "----------------------------------------"
    echo "Processing: $video_path"
    echo "Output directory: $video_out_dir"
    echo "Started at: $(date)"
    
    # Build command
    local cmd="python \"$PYTHON_SCRIPT\""
    cmd="$cmd \"$video_path\""
    cmd="$cmd \"$WEIGHTS\""
    cmd="$cmd \"$video_out_dir\""
    cmd="$cmd --class_id $CLASS_ID"
    cmd="$cmd --conf_thresh $CONF_THRESH"
    cmd="$cmd --line_pos $LINE_POS"
    cmd="$cmd --imgsz $IMGSZ_W $IMGSZ_H"
    cmd="$cmd --tracker \"$TRACKER\""
    cmd="$cmd --move_right \"$MOVE_RIGHT\""
    cmd="$cmd --move_left \"$MOVE_LEFT\""
    
    if [[ "$SAVE_VIDEO" == true ]]; then
        cmd="$cmd --save_video"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN - Would execute: $cmd"
        return 0
    fi
    
    # Execute command
    if eval "$cmd"; then
        echo "✅ Successfully processed: $video_basename"
        echo "Completed at: $(date)"
        return 0
    else
        echo "❌ Failed to process: $video_basename"
        echo "Failed at: $(date)"
        return 1
    fi
}

# Read video list and process each video
echo "========================================"
echo "Batch Fish Counter Processing"
echo "========================================"
echo "Video list file: $VIDEO_LIST_FILE"
echo "Model weights: $WEIGHTS"
echo "Base output directory: $OUT_DIR"
echo "Save annotated videos: $SAVE_VIDEO"
echo "Continue on error: $CONTINUE_ON_ERROR"

if [[ "$DRY_RUN" == true ]]; then
    echo "🔍 DRY RUN MODE - No actual processing will occur"
fi

echo ""

# Count total videos
total_videos=$(wc -l < "$VIDEO_LIST_FILE" | tr -d ' ')
current_video=0
successful_videos=0
failed_videos=0
failed_list=()

echo "Total videos to process: $total_videos"
echo ""

# Process each video file
while IFS= read -r video_path || [[ -n "$video_path" ]]; do
    # Skip empty lines and comments
    if [[ -z "$video_path" ]] || [[ "$video_path" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Remove leading/trailing whitespace
    video_path=$(echo "$video_path" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    if [[ -z "$video_path" ]]; then
        continue
    fi
    
    current_video=$((current_video + 1))
    
    echo "[$current_video/$total_videos] Processing video..."
    
    # Check if video file exists
    if [[ ! -f "$video_path" ]]; then
        echo "⚠️  Warning: Video file not found: $video_path"
        if [[ "$CONTINUE_ON_ERROR" == false ]]; then
            echo "Stopping due to missing file. Use --continue-on-error to skip missing files."
            exit 1
        else
            failed_videos=$((failed_videos + 1))
            failed_list+=("$video_path (file not found)")
            continue
        fi
    fi
    
    # Get video basename and create output directory
    video_basename=$(get_basename "$video_path")
    video_out_dir="$OUT_DIR/$video_basename"
    
    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$video_out_dir"
    fi
    
    # Process the video
    if process_video "$video_path" "$video_basename" "$video_out_dir"; then
        successful_videos=$((successful_videos + 1))
    else
        failed_videos=$((failed_videos + 1))
        failed_list+=("$video_path")
        
        if [[ "$CONTINUE_ON_ERROR" == false ]]; then
            echo "Stopping due to processing error. Use --continue-on-error to continue with remaining videos."
            exit 1
        fi
    fi
    
    echo ""
    
done < "$VIDEO_LIST_FILE"

# Final summary
echo "========================================"
echo "Batch Processing Complete"
echo "========================================"
echo "Total videos: $total_videos"
echo "Successful: $successful_videos"
echo "Failed: $failed_videos"

if [[ ${#failed_list[@]} -gt 0 ]]; then
    echo ""
    echo "Failed videos:"
    for failed_video in "${failed_list[@]}"; do
        echo "  - $failed_video"
    done
fi

echo ""
echo "Results saved in: $OUT_DIR"
echo "Finished at: $(date)"

# Exit with appropriate code
if [[ $failed_videos -gt 0 ]]; then
    exit 1
else
    exit 0
fi