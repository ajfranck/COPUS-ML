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
    ):

        self.raw_video_dir = Path(raw_video_dir)
        self.processed_dir = Path(processed_dir)
        self.splice_duration = splice_duration
        self.buffer_duration = buffer_duration
        self.total_segment_duration = splice_duration + (2 * buffer_duration)

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VideoProcessor initialized:")
        logger.info(f"  Raw video directory: {self.raw_video_dir}")
        logger.info(f"  Processed directory: {self.processed_dir}")
        logger.info(
            f"  Splice duration: {splice_duration}s with {buffer_duration}s buffers"
        )

    def get_video_duration(self, video_path: Path) -> float:
        """
        Gets duration of vid in seconds
        """

        try:
            probe = ffmpeg.probe(str(video_path))
            video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
            duration = float(probe["format"]["duration"])
            return duration
        except Exception as e:
            logger.error(f"can't get duration for {video_path}: {e}")
            return 0.0

    def get_lecture_folders(self) -> List[Path]:

        if not self.raw_video_dir.exists():
            logger.warning(f"raw direc not exist: {self.raw_video_dir}")
            return []

        lecture_folders = [
            f
            for f in self.raw_video_dir.iterdir()
            if f.is_dir() and f.name.isdigit() and len(f.name) == 8
        ]

        return sorted(lecture_folders)

    def get_mts_files(self, lecture_folder: Path) -> List[Path]:
        """
        Get all MTS files from lecture folder
        """

        mts_files = list(lecture_folder.glob("*.MTS"))
        mts_files.extend(list(lecture_folder.glob("*.mts")))

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
            (
                ffmpeg.input(concat_file, format="concat", safe=0)
                .output(str(output_path), c="copy")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            os.unlink(concat_file)

            logger.info(f"concatted to {output_path}")
            return True

        except ffmpeg.Error as e:
            logger.error(f"ffmpeg error: {e.stderr.decode()}")
            if os.path.exists(concat_file):
                os.unlink(concat_file)
            return False
        except Exception as e:
            logger.error(f"error: {e}")
            if os.path.exists(concat_file):
                os.unlink(concat_file)
            return False

    def splice_video(
        self, video_path: Path, output_folder: Path, lecture_date: str
    ) -> List[Path]:
        """
        Splices video into segments with buffer
        """

        output_folder.mkdir(parents=True, exist_ok=True)
        spliced_files = []

        try:
            duration = self.get_video_duration(video_path)
            if duration == 0:
                logger.error(f"cant determine video duration for {video_path}")
                return []

            logger.info(f"vid duration: {duration:.2f} secs")

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
                    f"[{buffered_start:.1f}s - {end_time:.1f}s]"
                )

                try:
                    (
                        ffmpeg.input(str(video_path), ss=buffered_start)
                        .output(
                            str(output_path),
                            t=segment_duration,
                            c="copy",
                            avoid_negative_ts="make_zero",
                        )
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )

                    spliced_files.append(output_path)

                except ffmpeg.Error as e:
                    logger.error(f"error w splice {i+1}: {e.stderr.decode()}")
                    continue

            logger.info(f"created {len(spliced_files)} splices")
            return spliced_files

        except Exception as e:
            logger.error(f"err splicing video {video_path}: {e}")
            return []

    def process_lecture_folder(self, lecture_folder: Path) -> bool:

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

        # multiple vids: concatenate then splice
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

        lecture_folders = self.get_lecture_folders()

        if not lecture_folders:
            logger.warning("no lecture folders found")
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
        logger.info(f"SUMMARY:")
        logger.info(f"  - total lectures processed: {len(lecture_folders)}")
        logger.info(f"  - total splices made: {total_splices}")
        logger.info(f"{'='*60}")


def main():
    processor = VideoProcessor(
        raw_video_dir="data/raw/videos",
        processed_dir="data/processed",
        splice_duration=120,
        buffer_duration=10,
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
        logger.info("\n all processed well")


if __name__ == "__main__":
    main()
