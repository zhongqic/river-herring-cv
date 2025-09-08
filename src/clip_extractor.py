import os
import re
import subprocess
import argparse
from datetime import datetime, timedelta

def parse_filename(filename):
    pattern = r"([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(\d+)\.mp4"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match the expected format")
    
    video_id, date, time, ms = match.groups()
    return video_id, date, time, ms

def format_filename(river, date, time, ms):
    return f"{river}_{date}_{time}_{ms}.mp4"

def convert_mmss_to_seconds(mmss):
    minutes, seconds = map(int, mmss.split(':'))
    return minutes * 60 + seconds

def convert_seconds_to_hhmmss(seconds):
    return str(timedelta(seconds=seconds))

def main(input_file, river, start_time_mmss, end_time_mmss, output_dir):
    # Parse input file name
    video_id, date, time, ms = parse_filename(os.path.basename(input_file))
    
    # Convert mm:ss to seconds
    start_time_seconds = convert_mmss_to_seconds(start_time_mmss)
    end_time_seconds = convert_mmss_to_seconds(end_time_mmss)
    
    # Calculate new filename components
    start_time_obj = datetime.strptime(f"{time}.{ms}", "%H-%M-%S.%f")
    new_start_time_obj = start_time_obj + timedelta(seconds=start_time_seconds)
    new_time = new_start_time_obj.strftime("%H-%M-%S")
    new_ms = new_start_time_obj.strftime("%f")[:3]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, format_filename(river, date, new_time, new_ms))
    
    # Construct and run ffmpeg command
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_file,
        '-ss', convert_seconds_to_hhmmss(start_time_seconds),
        '-to', convert_seconds_to_hhmmss(end_time_seconds),
        '-c', 'copy',
        output_file
    ]
    
    subprocess.run(ffmpeg_command, check=True)
    print(f"Extracted clip saved as: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a clip from a video using ffmpeg.")
    parser.add_argument("--input_file", required=True, help="Path to the input video file")
    parser.add_argument("--river", required=True, help="River name prefix for output file")
    parser.add_argument("--start", required=True, help="Start time (MM:SS)")
    parser.add_argument("--end", required=True, help="End time (MM:SS)")
    parser.add_argument("--output_dir", default=".", help="Directory where the clip will be saved (default: current directory)")
    
    args = parser.parse_args()
    
    main(args.input_file, args.river, args.start, args.end, args.output_dir)