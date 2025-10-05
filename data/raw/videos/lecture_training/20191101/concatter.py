import os
import subprocess
import re

def concatenate_videos_ffmpeg():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    files = os.listdir(script_dir)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    video_files = [f for f in files if f.lower().endswith(video_extensions)]
    
    def get_number(filename):
        match = re.search(r'^(\d+)', filename)
        return int(match.group(1)) if match else float('inf')
    
    video_files.sort(key=get_number)
    
    if not video_files:
        return
    
    print(f"Found {len(video_files)} video files:")
    for f in video_files:
        print(f"  - {f}")
    
    filelist_path = os.path.join(script_dir, 'filelist.txt')
    with open(filelist_path, 'w') as f:
        for video in video_files:
            full_path = os.path.join(script_dir, video)
            f.write(f"file '{full_path}'\n")
    
    print("Concatting") 
    output_path = os.path.join(script_dir, 'concatenated_output.mp4')
    subprocess.run([
        'ffmpeg', '-f', 'concat', '-safe', '0', 
        '-i', filelist_path, '-c', 'copy', 
        output_path
    ])
    
    os.remove(filelist_path)
    print("output: concatenated_output.mp4")

if __name__ == "__main__":
    concatenate_videos_ffmpeg()