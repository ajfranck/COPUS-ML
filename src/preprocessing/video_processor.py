import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from datetime import timedelta
import shutil
import tempfile
import ffmpeg


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VideoProcessor:

    def __init__(
        self,
        raw_video_dir: str = "data/raw/videos",
        processed_dir: str = "data/processed",
        splice_duration: int = 120,
        buffer_duration: int = 10,
        target_fps: Optional[float] = None,
    ):
        script_dir = Path(__file__).parent.parent.parent
        
        self.raw_video_dir = script_dir / raw_video_dir
        self.processed_dir = script_dir / processed_dir
        
        self.splice_duration = splice_duration
        self.buffer_duration = buffer_duration
        self.target_fps = target_fps
        self.total_segment_duration = splice_duration + (2 * buffer_duration)

        self.raw_video_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        copus_scores_dir = script_dir / "data" / "raw" / "copus_scores"
        copus_scores_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VideoProcessor init:")
        logger.info(f"  Raw video directory: {self.raw_video_dir}")
        logger.info(f"  Processed directory: {self.processed_dir}")
        logger.info(
            f"  Splice duration: {splice_duration}s with {buffer_duration}s buffers"
        )
        if target_fps:
            logger.info(f"  Target FPS: {target_fps}")
        else:
            logger.info("  Target FPS: Keep original")

    def get_video_info(self, video_path: Path) -> Tuple[float, float]:
        """
        Gets duration and FPS of video
        """
        try:
            probe = ffmpeg.probe(str(video_path))
            video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
            duration = float(probe["format"]["duration"])

            fps = None
            if "r_frame_rate" in video_info:
                fps_str = video_info["r_frame_rate"]
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else None
                else:
                    fps = float(fps_str)

            # fallback to avg_frame_rate if r_frame_rate not available
            if fps is None or fps == 0:
                if "avg_frame_rate" in video_info:
                    fps_str = video_info["avg_frame_rate"]
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        fps = float(num) / float(den) if float(den) != 0 else 30.0
                    else:
                        fps = float(fps_str)
                else:
                    fps = 30.0

            logger.info(f"vid info - duration: {duration:.2f}s, fps: {fps:.2f}")
            return duration, fps

        except Exception as e:
            logger.error(f"can't get video info for {video_path}: {e}")
            return 0.0, 30.0

    def get_video_duration(self, video_path: Path) -> float:

        duration, _ = self.get_video_info(video_path)
        return duration

    def get_lecture_folders(self) -> List[Path]:

        if not self.raw_video_dir.exists():
            logger.warning(f"vid direct not exist: {self.raw_video_dir}")
            logger.info("creating...")
            self.raw_video_dir.mkdir(parents=True, exist_ok=True)
            return []

        lecture_folders = [
            f
            for f in self.raw_video_dir.iterdir()
            if f.is_dir() and f.name.isdigit() and len(f.name) == 8
        ]
        
        if not lecture_folders:
            logger.info("no folders found: expected folder structure: data/raw/videos/YYYYMMDD/")

        return sorted(lecture_folders)

    def get_mts_files(self, lecture_folder: Path) -> List[Path]:

        mts_files = []
        
        for pattern in ["*.MTS", "*.mts"]:
            for file in lecture_folder.glob(pattern):
                if not file.name.startswith('.') and not file.name.startswith('._'):
                    mts_files.append(file)
        
        mts_files = sorted(set(mts_files), key=lambda x: x.name)
        
        return mts_files

    def concatenate_videos(self, video_files: List[Path], output_path: Path) -> bool:
        """
        Concats multiple video files into one
        """

        if not video_files:
            logger.warning("no video files to concatenate")
            return False

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                for video_file in video_files:
                    f.write(f"file '{video_file.absolute()}'\n")
                concat_file = f.name

            logger.info(f"concatting {len(video_files)} vids")

            # Build ffmpeg command for concatenation
            input_stream = ffmpeg.input(concat_file, format="concat", safe=0)
            output_args = {}

            if self.target_fps:
                output_args["r"] = self.target_fps
                output_args["c:v"] = "libx264"
                output_args["preset"] = "medium"
                output_args["crf"] = "23"
                output_args["c:a"] = "aac"
                output_args["b:a"] = "192k"
                logger.info(f"Setting FPS to {self.target_fps} during concatenation")
            else:
                output_args["c:v"] = "copy"
                # Still need to convert audio for MP4 compatibility
                output_args["c:a"] = "aac"
                output_args["b:a"] = "192k"

            (
                input_stream.output(str(output_path), **output_args)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            os.unlink(concat_file)

            logger.info(f"concatted to {output_path}")
            return True

        except Exception as e:
            error_msg = str(e)
            # Try to extract stderr from various exception types
            if hasattr(e, 'stderr'):
                if isinstance(e.stderr, bytes):
                    error_msg = e.stderr.decode('utf-8', errors='ignore')
                else:
                    error_msg = str(e.stderr)
            
            logger.error(f"ffmpeg error during concatenation: {error_msg}")
            if os.path.exists(concat_file):
                os.unlink(concat_file)
            return False

    def splice_video(
        self, video_path: Path, output_folder: Path, lecture_date: str
    ) -> List[Path]:
        """
        Splice video into segments with buffers
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        spliced_files = []

        try:
            duration, original_fps = self.get_video_info(video_path)
            if duration == 0:
                logger.error(f"cant determine video duration for {video_path}")
                return []

            effective_fps = self.target_fps if self.target_fps else original_fps
            logger.info(
                f"vid duration: {duration:.2f} secs, using FPS: {effective_fps:.2f}"
            )

            num_splices = int(duration / self.splice_duration) + (
                1 if duration % self.splice_duration > 0 else 0
            )

            logger.info(f"cooking {num_splices} splices")

            for i in range(num_splices):
                # calculate start time
                start_time = i * self.splice_duration

                # add buffer before
                buffered_start = max(0, start_time - self.buffer_duration)

                # calcs end time with buffer after
                end_time = min(
                    start_time + self.splice_duration + self.buffer_duration, duration
                )

                # calcs actual duration
                segment_duration = end_time - buffered_start

                # filename: lecturedate_splice_number.mp4
                output_filename = f"{lecture_date}_{i+1:03d}.mp4"
                output_path = output_folder / output_filename

                logger.info(
                    f"cooking splice {i+1}/{num_splices}: {output_filename} "
                    f"[{buffered_start:.1f}s - {end_time:.1f}s] @ {effective_fps:.1f}fps"
                )

                try:
                    input_stream = ffmpeg.input(str(video_path), ss=buffered_start)

                    output_args = {
                        "t": segment_duration,
                        "avoid_negative_ts": "make_zero",
                    }

                    if self.target_fps:
                        output_args["r"] = self.target_fps
                        output_args["c:v"] = "libx264"
                        output_args["preset"] = "medium"
                        output_args["crf"] = "23"
                        # Convert audio to AAC for MP4 compatibility
                        output_args["c:a"] = "aac"
                        output_args["b:a"] = "192k"
                    else:
                        output_args["c:v"] = "copy"
                        # Still need to convert audio for MP4 compatibility
                        output_args["c:a"] = "aac"
                        output_args["b:a"] = "192k"

                    (
                        input_stream.output(str(output_path), **output_args)
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )

                    spliced_files.append(output_path)

                except Exception as e:
                    error_msg = str(e)
                    # Try to extract stderr from various exception types
                    if hasattr(e, 'stderr'):
                        if isinstance(e.stderr, bytes):
                            error_msg = e.stderr.decode('utf-8', errors='ignore')
                        else:
                            error_msg = str(e.stderr)
                    
                    logger.error(f"ffmpeg error with splice {i+1}: {error_msg}")
                    continue

            logger.info(f"created {len(spliced_files)} splices")
            return spliced_files

        except Exception as e:
            logger.error(f"err splicing video {video_path}: {e}")
            return []

    def process_lecture_folder(self, lecture_folder: Path) -> bool:
        """
        Process a single lecture folder
        """
        lecture_date = lecture_folder.name
        logger.info(f"\nprocessing: {lecture_date}")
        logger.info("=" * 50)

        mts_files = self.get_mts_files(lecture_folder)

        if not mts_files:
            logger.warning(f"no mts in {lecture_folder}")
            return False

        logger.info(f"{len(mts_files)} mts files: {[f.name for f in mts_files]}")

        output_folder = self.processed_dir / lecture_date

        if len(mts_files) == 1:
            logger.info("splicing single vid file")
            spliced_files = self.splice_video(mts_files[0], output_folder, lecture_date)
            return len(spliced_files) > 0

        logger.info(f"multiple vid files ({len(mts_files)}), concat first")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_concat:
            temp_concat_path = Path(temp_concat.name)

        try:
            if not self.concatenate_videos(mts_files, temp_concat_path):
                logger.error("concat failed")
                return False

            spliced_files = self.splice_video(
                temp_concat_path, output_folder, lecture_date
            )

            if temp_concat_path.exists():
                temp_concat_path.unlink()

            return len(spliced_files) > 0

        except Exception as e:
            logger.error(f"error processing lecture folder: {e}")
            if temp_concat_path.exists():
                temp_concat_path.unlink()
            return False

    def process_all_lectures(self) -> Tuple[int, int]:
        """
        Process all lecture folders
        """
        lecture_folders = self.get_lecture_folders()

        if not lecture_folders:
            logger.warning("No lecture folders")
            logger.info("1. create folders w/ 8-digit names (YYYYMMDD format)")
            logger.info("2. put mts vds in  those folders")
            return 0, 0

        logger.info(f"\n{len(lecture_folders)} lecture folders")
        logger.info(f"lectures: {[f.name for f in lecture_folders]}")

        successful = 0
        failed = 0

        for i, lecture_folder in enumerate(lecture_folders, 1):
            logger.info(f"\n{'='*60}")
            logger.info(
                f"doing lecture {i}/{len(lecture_folders)}: {lecture_folder.name}"
            )
            logger.info(f"{'='*60}")

            if self.process_lecture_folder(lecture_folder):
                successful += 1
                logger.info(f"done w {lecture_folder.name}")
            else:
                failed += 1
                logger.error(f"failed processing {lecture_folder.name}")

        return successful, failed

    def verify_processing(self) -> None:

        logger.info("report")

        lecture_folders = sorted(
            [f for f in self.processed_dir.iterdir() if f.is_dir()]
        )

        if not lecture_folders:
            logger.warning("no processed lecture vids found")
            return

        total_splices = 0

        for folder in lecture_folders:
            splices = list(folder.glob("*.mp4"))
            total_splices += len(splices)

            logger.info(f"\nlecture {folder.name}:")
            logger.info(f"  - num splices: {len(splices)}")

            if splices:
                first_splice = sorted(splices)[0].name
                last_splice = sorted(splices)[-1].name
                logger.info(f"  - first splice: {first_splice}")
                logger.info(f"  - last splice: {last_splice}")

                total_duration = len(splices) * self.splice_duration
                logger.info(f"  - total duration: {timedelta(seconds=total_duration)}")

        logger.info(f"\n{'='*60}")
        logger.info(f"summary:")
        logger.info(f"  - total lectures processed: {len(lecture_folders)}")
        logger.info(f"  - total splices made: {total_splices}")
        if self.target_fps:
            logger.info(f"  - target FPS: {self.target_fps}")
        logger.info(f"{'='*60}")


def check_dependencies():
    import subprocess
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("ffmpeg not working properly")
    except FileNotFoundError:
        logger.error("ffmpeg not in system PATH")
        logger.error("install ffmpeg with choco or brew or sudo:")
        return False
    except Exception as e:
        logger.error(f"err checking ffmpeg: {e}")
        return False
    
    return True


def main():
    script_dir = Path(__file__).parent
    
    project_root = script_dir.parent.parent
    os.chdir(project_root)
    
    logger.info(f"cwd: {os.getcwd()}")
    
    if not check_dependencies():
        logger.error("\ninstall dependenceies")
        sys.exit(1)
    
    processor = VideoProcessor(
        raw_video_dir="data/raw/videos/lecture_training",
        processed_dir="data/processed/training/lecturing/",
        # splice_duration=120,
        # buffer_duration=10,
        splice_duration=10,
        buffer_duration=0,
        target_fps=3.0,
    )

    logger.info("\n" + "=" * 60)
    logger.info("vid processing")
    logger.info("=" * 60)

    successful, failed = processor.process_all_lectures()

    logger.info("\n" + "=" * 60)
    logger.info("...done")
    logger.info(f"  - success: {successful}")
    logger.info(f"  - failed: {failed}")
    logger.info("=" * 60)

    processor.verify_processing()

    if failed > 0:
        logger.warning(f"\n{failed} lectures failed")
        sys.exit(1)
    else:
        logger.info("\n all processed well" if successful > 0 else "\nNo lectures to process")


if __name__ == "__main__":
    main()