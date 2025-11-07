import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import logging
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download, snapshot_download
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
import math
import argparse
from collections import defaultdict
import time
import tempfile
import re

try:
    import ffmpeg

    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logging.warning(
        "ffmpeg-python not installed. MTS conversion will not be available."
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

LECTURE_FPS = 3  # Input video is 3fps
WINDOW_DURATION = 10  # 10 second windows
WINDOW_FRAMES = LECTURE_FPS * WINDOW_DURATION  # 30 frames per window
WINDOW_STEP = 5  # Step size in seconds (50% overlap)
AGGREGATION_INTERVAL = 120  # 2 minutes in seconds

MAX_NUM_FRAMES = 15
MAX_NUM_PACKING = 2
TIME_SCALE = 0.1

COPUS_ACTIONS = {
    "student_listening": 0,
    "student_individual_thinking": 1,
    "student_clicker_group": 2,
    "student_worksheet_group": 3,
    "student_other_group": 4,
    "student_answer_question": 5,
    "student_ask_question": 6,
    "student_whole_class_discussion": 7,
    "student_prediction": 8,
    "student_presentation": 9,
    "student_test_quiz": 10,
    "student_waiting": 11,
    "student_other": 12,
    "instructor_lecturing": 13,
    "instructor_real_time_writing": 14,
    "instructor_follow_up": 15,
    "instructor_posing_question": 16,
    "instructor_clicker_question": 17,
    "instructor_answering_question": 18,
    "instructor_moving_guiding": 19,
    "instructor_one_on_one": 20,
    "instructor_demo_video": 21,
    "instructor_administration": 22,
    "instructor_waiting": 23,
}

ACTIONS_REVERSE = {v: k for k, v in COPUS_ACTIONS.items()}

COPUS_LABELS = {
    "student_listening": "L - Listening/Taking Notes",
    "student_individual_thinking": "Ind - Individual Thinking",
    "student_clicker_group": "CG - Clicker Group Discussion",
    "student_worksheet_group": "WG - Worksheet Group Work",
    "student_other_group": "OG - Other Group Activity",
    "student_answer_question": "AnQ - Answering Question",
    "student_ask_question": "SQ - Asking Question",
    "student_whole_class_discussion": "WC - Whole Class Discussion",
    "student_prediction": "Prd - Making Prediction",
    "student_presentation": "SP - Student Presentation",
    "student_test_quiz": "TQ - Test/Quiz",
    "student_waiting": "W - Waiting",
    "student_other": "O - Other",
    "instructor_lecturing": "Lec - Lecturing",
    "instructor_real_time_writing": "RtW - Real-time Writing",
    "instructor_follow_up": "FUp - Follow-up/Feedback",
    "instructor_posing_question": "PQ - Posing Question",
    "instructor_clicker_question": "CQ - Clicker Question",
    "instructor_answering_question": "AnQ - Answering Questions",
    "instructor_moving_guiding": "MG - Moving/Guiding",
    "instructor_one_on_one": "1o1 - One-on-One",
    "instructor_demo_video": "D/V - Demo/Video",
    "instructor_administration": "Adm - Administration",
    "instructor_waiting": "W - Waiting",
}


def map_to_nearest_scale(values, scale):
    """Map values to nearest scale point"""
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    """Group array into chunks of specified size"""
    return [arr[i : i + size] for i in range(0, len(arr), size)]


def encode_video_window(frames_array, fps=3):
    """
    Encode a window of frames for the model

    Args:
        frames_array: numpy array of frames
        fps: frames per second

    Returns:
        frames: List of PIL Images
        temporal_ids: List of temporal ID groups
    """
    try:
        num_frames = len(frames_array)
        duration = num_frames / fps

        frames = [
            Image.fromarray(frame.astype("uint8")).convert("RGB")
            for frame in frames_array
        ]

        frame_idx = np.arange(num_frames)
        frame_idx_ts = frame_idx / fps
        scale = np.arange(0, duration, TIME_SCALE)

        if len(scale) == 0:
            scale = np.array([0])

        frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
        frame_ts_id = frame_ts_id.astype(np.int32)

        if num_frames <= MAX_NUM_FRAMES:
            packing_nums = 1
        else:
            packing_nums = min(math.ceil(num_frames / MAX_NUM_FRAMES), MAX_NUM_PACKING)

        frame_ts_id_group = group_array(frame_ts_id, packing_nums)

        return frames, frame_ts_id_group

    except Exception as e:
        logger.error(f"Error encoding video window: {e}")
        return [], []


class FullLectureEvaluator:

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        base_model: str = "openbmb/MiniCPM-V-4_5",
        hf_model_repo: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        logger.info(f"Initializing Full Lecture Evaluator on device: {self.device}")

        self.checkpoint_path = None
        self.classifier = None
        self.hf_classifier_path = None
        
        # Priority 1: Try HuggingFace Hub for classifier only
        if hf_model_repo:
            try:
                logger.info(f"Downloading classifier from HuggingFace Hub: {hf_model_repo}")
                # Download just the classifier file
                cache_dir = Path.home() / ".cache" / "copus-ml"
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Try to download classifier.pt
                try:
                    classifier_path = hf_hub_download(
                        repo_id=hf_model_repo,
                        filename="classifier.pt",
                        cache_dir=str(cache_dir),
                        resume_download=True
                    )
                    self.hf_classifier_path = Path(classifier_path)
                    logger.info(f"Classifier downloaded: {self.hf_classifier_path.name}")
                except Exception as e:
                    logger.warning(f"Could not download classifier.pt: {e}")
                    logger.info("Trying full checkpoint instead")
                    # Fallback: try downloading the full snapshot
                    model_path = snapshot_download(
                        repo_id=hf_model_repo,
                        cache_dir=str(cache_dir),
                        resume_download=True
                    )
                    self.checkpoint_path = Path(model_path)
                    logger.info(f"Checkpoint downloaded: {self.checkpoint_path}")
                
            except Exception as e:
                logger.warning(f"Failed to download from HuggingFace: {e}")
                logger.info("Falling back to local checkpoint or base model")

        # Priority 2: Use provided local checkpoint path
        if self.checkpoint_path is None and checkpoint_path and Path(checkpoint_path).exists():
            self.checkpoint_path = Path(checkpoint_path)
            logger.info(f"Loading from local path: {checkpoint_path}")
        
        # Priority 3: Search for local checkpoint in default directory
        if self.checkpoint_path is None and self.hf_classifier_path is None:
            models_dir = Path("src/models")
            if models_dir.exists():
                checkpoints = list(models_dir.glob("copus_model_*"))
                if checkpoints:
                    self.checkpoint_path = sorted(checkpoints)[-1]
                    logger.info(f"Found local checkpoint: {self.checkpoint_path}")

        # Load the base model
        # If we have a full checkpoint, load from there; otherwise use base model
        if self.checkpoint_path and not self.hf_classifier_path:
            # Try to load as full model (legacy checkpoints)
            try:
                logger.info("Loading model from checkpoint")
                self.model = AutoModel.from_pretrained(
                    str(self.checkpoint_path),
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.checkpoint_path), trust_remote_code=True
                )
                logger.info("Model loaded from checkpoint")
            except Exception as e:
                logger.warning(f"Could not load model from checkpoint: {e}")
                logger.info("Loading base model instead")
                self.model = AutoModel.from_pretrained(
                    base_model,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model, trust_remote_code=True
                )
                logger.info(f"Base model loaded: {base_model}")
        else:
            # Load base model (this is the standard path for HF classifier downloads)
            logger.info(f"Loading base model: {base_model}")
            self.model = AutoModel.from_pretrained(
                base_model,
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True
            )
            logger.info("Base model loaded")

        self.model = self.model.eval().to(self.device)

        # Load classifier
        classifier_loaded = False
        
        # Try HuggingFace classifier first
        if self.hf_classifier_path and self.hf_classifier_path.exists():
            try:
                self.load_classifier(self.hf_classifier_path)
                logger.info("HuggingFace classifier loaded")
                classifier_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace classifier: {e}")
        
        # Try checkpoint classifier
        if not classifier_loaded and self.checkpoint_path:
            classifier_path = self.checkpoint_path / "classifier.pt"
            if classifier_path.exists():
                try:
                    self.load_classifier(classifier_path)
                    logger.info("Checkpoint classifier loaded")
                    classifier_loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint classifier: {e}")
        
        if not classifier_loaded:
            logger.warning("No fine-tuned classifier loaded - using base model only")

    def load_classifier(self, classifier_path):
        """Load classifier from checkpoint file"""
        checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)

        hidden_size = (
            self.model.config.hidden_size
            if hasattr(self.model.config, "hidden_size")
            else 4096
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(COPUS_ACTIONS)),
        ).to(self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "classifier_state_dict" in checkpoint:
                # Standard checkpoint format
                self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
                epoch = checkpoint.get('epoch', 'unknown')
                logger.info(f"Classifier loaded from epoch {epoch}")
            elif "model_state_dict" in checkpoint:
                # Alternative format
                self.classifier.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Classifier loaded (model_state_dict format)")
            else:
                # Assume the dict itself is the state dict
                self.classifier.load_state_dict(checkpoint)
                logger.info("Classifier loaded (direct state_dict format)")
        else:
            # Direct state dict (not wrapped in a dictionary)
            self.classifier.load_state_dict(checkpoint)
            logger.info("Classifier loaded (unwrapped state_dict)")

        self.classifier.eval()


    def convert_mts_to_mp4(self, mts_path: Path) -> Optional[Path]:
        """
        Convert MTS file to temporary MP4 file for processing

        Args:
            mts_path: Path to MTS file

        Returns:
            Path to temporary MP4 file, or None if conversion failed

        Note: I did this because I thought necessary but may just wanna convert to mp4 manually beforehand
        """
        if not FFMPEG_AVAILABLE:
            logger.warning("ffmpeg-python not available for MTS conversion")
            return None

        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_path = Path(temp_file.name)
            temp_file.close()

            logger.info(f"Converting {mts_path.name} to MP4")

            (
                ffmpeg.input(str(mts_path))
                .output(
                    str(temp_path),
                    vcodec="libx264",
                    preset="ultrafast",
                    crf=23,
                    acodec="aac",
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            logger.info(f"Converted to: {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Error converting MTS file: {e}")
            if "temp_path" in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            return None

    def parse_action_classifications(self, model_response: str) -> Dict[str, float]:
        """
        Parse model  response to hopefully get action classifications and confidence scores
        
        Returns:
            Dictionary mapping action codes to confidence scores (0.0-1.0)
        """
        predictions = {}
        
        confidence_map = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3
        }
        
        if "DETECTED ACTIONS:" in model_response:
            actions_section = model_response.split("DETECTED ACTIONS:")[1]
            
            lines = actions_section.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('Example:'):
                    continue
                    
                # try and match: action_code: confidence - justification
                for action in COPUS_ACTIONS.keys():
                    if line.startswith(action + ":"):
                        confidence_str = "medium"  # default
                        for conf_word in ["high", "medium", "low"]:
                            if conf_word in line.lower():
                                confidence_str = conf_word
                                break
                        
                        predictions[action] = confidence_map[confidence_str]
                        break
        
        # some fallback in case it fails (unlikely)
        if not predictions:
            # look for common patterns in the response
            response_lower = model_response.lower()
            
            if "instructor" in response_lower and ("lecturing" in response_lower or "explaining" in response_lower or "presenting" in response_lower):
                predictions["instructor_lecturing"] = 0.5
            
            if "students" in response_lower and ("listening" in response_lower or "watching" in response_lower or "taking notes" in response_lower):
                predictions["student_listening"] = 0.5
            
            if "writing on" in response_lower and "board" in response_lower:
                predictions["instructor_real_time_writing"] = 0.5
            
            if "group" in response_lower and ("discussion" in response_lower or "working together" in response_lower):
                predictions["student_other_group"] = 0.5
            
            # if nothinhg default to most common scenario
            if not predictions:
                predictions["instructor_lecturing"] = 0.3
                predictions["student_listening"] = 0.3
        
        return predictions

    def evaluate_window(self, frames: List[Image.Image], temporal_ids: List) -> Dict:
        """
        Evaluate a single window of frames

        Returns:
            Dictionary with detected actions and confidence scores
        """
        # Create the classification prompt
        copus_actions_list = "\n".join([f"- {action}: {COPUS_LABELS[action]}" for action in COPUS_ACTIONS.keys()])
        
        classification_prompt = f"""Analyze this classroom video and identify ALL COPUS actions that are currently occurring.

        COPUS Actions to identify:
        {copus_actions_list}

        Instructions:
        1. Watch the video segment carefully
        2. Identify ALL actions that are happening (there can be multiple simultaneous actions)
        3. For each action you identify, provide:
        - The exact action code from the list above
        - Your confidence level (high, medium, or low)
        - A brief justification

        Format your response as:
        DETECTED ACTIONS:
        [action_code_1]: [confidence_level] - [brief justification]
        [action_code_2]: [confidence_level] - [brief justification]
        ...

        Example:
        DETECTED ACTIONS:
        instructor_lecturing: high - The instructor is standing at the board explaining concepts
        student_listening: high - Students are sitting and watching the instructor
        student_individual_thinking: medium - Some students appear to be working on problems individually

        Be thorough and identify ALL observable actions. Multiple actions often occur simultaneously."""

        msgs = [{"role": "user", "content": frames + [classification_prompt]}]

        try:
            with torch.no_grad():
                answer = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    use_image_id=False,
                    max_slice_nums=1,
                    temporal_ids=temporal_ids,
                )

            predictions = self.parse_action_classifications(answer)

            return {"description": answer, "predictions": predictions}

        except Exception as e:
            logger.error(f"Error evaluating window: {e}")
            return {"error": str(e), "predictions": {}}

    def evaluate_full_lecture(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        convert_mts: bool = True,
    ) -> Dict:
        """
        Evaluate a full lecture video using sliding windows

        Args:
            video_path: Path to the full lecture video (3fps)
            output_path: Optional path to save results
            convert_mts: If True, convert .MTS files to .MP4 temporarily

        Returns:
            Dictionary with aggregated results in 2-minute intervals
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return {"error": "Video file not found"}

        logger.info(f"Evaluating lecture: {video_path.name}")
        logger.info("-" * 60)

        temp_video_path = None
        original_path = video_path

        if video_path.suffix.upper() == ".MTS" and convert_mts:
            logger.info(
                "Detected .MTS file. Converting to temporary .MP4 for processing..."
            )
            temp_video_path = self.convert_mts_to_mp4(video_path)
            if temp_video_path:
                video_path = temp_video_path
                logger.info(f"Using temp file: {temp_video_path}")
            else:
                logger.warning("MTS conversion failed, attempting direct processing...")

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            duration = total_frames / fps

            logger.info("Video info:")
            logger.info(
                f"  - Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)"
            )
            logger.info(f"  - FPS: {fps:.1f}")
            logger.info(f"  - Total frames: {total_frames}")

            if abs(fps - LECTURE_FPS) > 0.5:
                logger.warning(f"Expected {LECTURE_FPS}fps, got {fps:.1f}fps")

            num_windows = int((duration - WINDOW_DURATION) / WINDOW_STEP) + 1
            logger.info(f"  - Number of windows to process: {num_windows}")

            window_results = []
            start_time = time.time()

            for window_idx in range(num_windows):
                window_start_sec = window_idx * WINDOW_STEP
                window_end_sec = window_start_sec + WINDOW_DURATION

                start_frame = int(window_start_sec * fps)
                end_frame = min(int(window_end_sec * fps), total_frames)

                if end_frame - start_frame < 10:
                    continue

                frame_indices = list(range(start_frame, end_frame))
                frames_array = vr.get_batch(frame_indices).asnumpy()

                frames, temporal_ids = encode_video_window(frames_array, fps)

                if not frames:
                    logger.warning(
                        f"Failed to encode window {window_idx+1}/{num_windows}"
                    )
                    continue

                logger.info(
                    f"Processing window {window_idx+1}/{num_windows} "
                    f"[{window_start_sec:.0f}s - {window_end_sec:.0f}s]"
                )

                result = self.evaluate_window(frames, temporal_ids)

                window_results.append(
                    {
                        "window_idx": window_idx,
                        "start_time": window_start_sec,
                        "end_time": window_end_sec,
                        "predictions": result.get("predictions", {}),
                        "raw_response": result.get("description", "")
                    }
                )

                if (window_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / (window_idx + 1)) * (num_windows - window_idx - 1)
                    logger.info(
                        f"  Progress: {window_idx+1}/{num_windows} windows "
                        f"(ETA: {eta/60:.1f} min)"
                    )

            aggregated_results = self.aggregate_results(window_results, duration)

            output = {
                "video_path": str(original_path),
                "video_info": {
                    "duration_seconds": duration,
                    "duration_minutes": duration / 60,
                    "fps": fps,
                    "total_frames": total_frames,
                },
                "processing_info": {
                    "window_duration": WINDOW_DURATION,
                    "window_step": WINDOW_STEP,
                    "total_windows": len(window_results),
                    "aggregation_interval": AGGREGATION_INTERVAL,
                },
                "intervals": aggregated_results,
                "timestamp": datetime.now().isoformat(),
            }

            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(output, f, indent=2)
                logger.info(f"Results saved to: {output_file}")

            self.print_summary(output)

            if temp_video_path and temp_video_path.exists():
                try:
                    temp_video_path.unlink()
                    logger.info(f"Cleaned up temp file: {temp_video_path}")
                except Exception as e:
                    logger.warning(f"Could not delete temporary file: {e}")

            return output

        except Exception as e:
            logger.error(f"Error processing video: {e}")

            if temp_video_path and temp_video_path.exists():
                try:
                    temp_video_path.unlink()
                except:
                    pass

            return {"error": str(e)}

    def aggregate_results(
        self, window_results: List[Dict], total_duration: float
    ) -> List[Dict]:
        """
        Aggregate window results into 2-minute intervals

        Returns:
            List of dictionaries, one per 2-minute interval
        """
        num_intervals = int(math.ceil(total_duration / AGGREGATION_INTERVAL))
        intervals = []

        for interval_idx in range(num_intervals):
            interval_start = interval_idx * AGGREGATION_INTERVAL
            interval_end = min(
                (interval_idx + 1) * AGGREGATION_INTERVAL, total_duration
            )

            overlapping_windows = []
            for window in window_results:
                if (
                    window["start_time"] < interval_end
                    and window["end_time"] > interval_start
                ):
                    overlapping_windows.append(window)

            actions_detected = defaultdict(float)
            for window in overlapping_windows:
                for action, confidence in window["predictions"].items():
                    actions_detected[action] = max(actions_detected[action], confidence)

            actions_binary = {}
            confidence_threshold = 0.3

            for action in COPUS_ACTIONS.keys():
                actions_binary[action] = (
                    actions_detected.get(action, 0) >= confidence_threshold
                )

            intervals.append(
                {
                    "interval_number": interval_idx + 1,
                    "start_time": interval_start,
                    "end_time": interval_end,
                    "start_time_str": str(timedelta(seconds=int(interval_start))),
                    "end_time_str": str(timedelta(seconds=int(interval_end))),
                    "actions": actions_binary,
                    "actions_with_confidence": dict(actions_detected),
                    "num_windows": len(overlapping_windows),
                }
            )

        return intervals

    def print_summary(self, results: Dict):
        logger.info("-" * 60)
        logger.info("Evaluation Summary")
        logger.info("-" * 60)

        if "error" in results:
            logger.error(f"Evaluation failed: {results['error']}")
            return

        video_info = results["video_info"]
        logger.info(f"Video: {Path(results['video_path']).name}")
        logger.info(f"Duration: {video_info['duration_minutes']:.1f} minutes")
        logger.info(f"Number of 2-minute intervals: {len(results['intervals'])}")

        action_counts = defaultdict(int)
        for interval in results["intervals"]:
            for action, present in interval["actions"].items():
                if present:
                    action_counts[action] += 1

        if action_counts:
            logger.info("Actions detected:")
            sorted_actions = sorted(
                action_counts.items(), key=lambda x: x[1], reverse=True
            )

            for action, count in sorted_actions[:10]:  # Top 10 actions
                percentage = (count / len(results["intervals"])) * 100
                readable = COPUS_LABELS.get(action, action)
                logger.info(
                    f"  - {readable}: {count}/{len(results['intervals'])} "
                    f"intervals ({percentage:.1f}%)"
                )

        logger.info("Activity timeline (first 10):")
        for interval in results["intervals"][:10]:
            active_actions = [
                action for action, present in interval["actions"].items() if present
            ]
            if active_actions:
                top_actions = active_actions[:3]
                action_str = ", ".join(
                    [COPUS_LABELS.get(a, a).split(" - ")[0] for a in top_actions]
                )
                logger.info(
                    f"  {interval['start_time_str']} - {interval['end_time_str']}: {action_str}"
                )

        logger.info("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate full COPUS lecture video")
    parser.add_argument(
        "video_path", type=str, help="Path to full lecture video (3fps, .mp4 or .mts)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output JSON file path for results",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--no-convert-mts",
        action="store_true",
        help="Attempt to process MTS files directly without conversion",
    )

    args = parser.parse_args()

    evaluator = FullLectureEvaluator(
        checkpoint_path=args.checkpoint, device=args.device, batch_size=args.batch_size
    )

    results = evaluator.evaluate_full_lecture(
        args.video_path, args.output, convert_mts=(not args.no_convert_mts)
    )

    if "error" not in results:
        logger.info(f"Evaluation complete. Results saved to: {args.output}")

        logger.info("Sample interval data (first):")
        if results["intervals"]:
            first_interval = results["intervals"][0]
            logger.info(
                f"  Time: {first_interval['start_time_str']} - {first_interval['end_time_str']}"
            )
            logger.info("  Actions:")
            for action, present in first_interval["actions"].items():
                if present:
                    logger.info(f"    - {COPUS_LABELS.get(action, action)}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # logger.info("Running in test mode...")
        # logger.info("Usage: python full_lecture_evaluation.py <video_path> --output <output.json>")
        # logger.info("\nSupported formats: .mp4, .mts, .avi, .mov")
        # logger.info("\nExample:")
        # logger.info("  python full_lecture_evaluation.py data/raw/videos/lecture.mp4 "
        #            "--output data/results/lecture_copus.json")
        # logger.info("\nFor MTS files:")
        # logger.info("  python full_lecture_evaluation.py data/raw/videos/00001.MTS "
        #            "--output data/results/lecture_copus.json")
        # logger.info("\nTo process MTS directly without conversion:")
        # logger.info("  python full_lecture_evaluation.py video.MTS --output output.json --no-convert-mts")

        test_videos = [
            Path("data/processed/training/lecture_full/20191101/full_lecture_1101.mp4"),
        ]

        for test_video in test_videos:
            if test_video.exists():
                logger.info(f"Found test video: {test_video}")
                evaluator = FullLectureEvaluator()
                results = evaluator.evaluate_full_lecture(
                    str(test_video), "data/results/test_lecture_copus.json"
                )
                break
        else:
            logger.info("No test video found. Please provide a video path")
    else:
        main()