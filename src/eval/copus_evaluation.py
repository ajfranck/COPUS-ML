import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
import math
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MAX_NUM_FRAMES = 15
MAX_NUM_PACKING = 2
TIME_SCALE = 0.1
TARGET_FPS = 10
VIDEO_DURATION = 7.0

COPUS_ACTIONS = {
    'student_question': 0,
    'student_answer': 1,
    'student_discussion': 2,
    'student_presentation': 3,
    'student_writing': 4,
    'student_listening': 5,
    'student_other': 6,
    'instructor_lecturing': 7,
    'instructor_writing': 8,
    'instructor_demonstrating': 9,
    'instructor_question': 10,
    'instructor_answer': 11,
    'instructor_moving': 12,
    'instructor_other': 13,
}

ACTIONS_REVERSE = {v: k for k, v in COPUS_ACTIONS.items()}


def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]


def encode_video(video_path, choose_fps=10, target_duration=7.0):
    """
    Args:
        video_path: Path to video file
        choose_fps: Target FPS for sampling
        target_duration: Target duration in seconds
    
    Returns:
        frames: List of PIL Images
        temporal_ids: List of temporal ID groups
    """
    def uniform_sample(l, n):
        if n >= len(l):
            return l
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        video_duration = len(vr) / fps
        
        logger.info(f"Video info - Duration: {video_duration:.2f}s, FPS: {fps:.2f}")
        
        target_frames = int(target_duration * choose_fps)
        
        if target_frames <= MAX_NUM_FRAMES:
            packing_nums = 1
            choose_frames = min(target_frames, len(vr))
        else:
            packing_nums = math.ceil(target_frames / MAX_NUM_FRAMES)
            packing_nums = min(packing_nums, MAX_NUM_PACKING)
            choose_frames = target_frames
        
        frame_idx = list(range(len(vr)))
        frame_idx = uniform_sample(frame_idx, choose_frames)
        frame_idx = np.array(frame_idx)
        
        logger.info(f"Sampling {len(frame_idx)} frames with packing_nums={packing_nums}")
        
        frames = vr.get_batch(frame_idx).asnumpy()
        frame_idx_ts = frame_idx / fps
        scale = np.arange(0, video_duration, TIME_SCALE)
        frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
        frame_ts_id = frame_ts_id.astype(np.int32)
        
        frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
        frame_ts_id_group = group_array(frame_ts_id, packing_nums)
        
        return frames, frame_ts_id_group
        
    except Exception as e:
        logger.error(f"Error encoding video {video_path}: {e}")
        return [], []


class COPUSEvaluator:
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        base_model: str = 'openbmb/MiniCPM-V-4_5',
        device: str = 'cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing evaluator on device: {self.device}")
        
        self.checkpoint_path = None
        self.classifier = None
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self.checkpoint_path = Path(checkpoint_path)
            logger.info(f"Loading fine-tuned model from: {checkpoint_path}")
        else:
            models_dir = Path("src/models")
            if models_dir.exists():
                checkpoints = list(models_dir.glob("copus_model_*"))
                if checkpoints:
                    self.checkpoint_path = sorted(checkpoints)[-1]
                    logger.info(f"Found checkpoint: {self.checkpoint_path}")
        
        if self.checkpoint_path:
            self.model = AutoModel.from_pretrained(
                str(self.checkpoint_path),
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.checkpoint_path),
                trust_remote_code=True
            )
            
            classifier_path = self.checkpoint_path / 'classifier.pt'
            if classifier_path.exists():
                self.load_classifier(classifier_path)
                logger.info("Loaded fine-tuned classifier")
        else:
            logger.info(f"Loading base model: {base_model}")
            self.model = AutoModel.from_pretrained(
                base_model,
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True
            )
        
        self.model = self.model.eval().to(self.device)
    
    def load_classifier(self, classifier_path):
        checkpoint = torch.load(classifier_path, map_location=self.device)
        
        hidden_size = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 4096
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(COPUS_ACTIONS))
        ).to(self.device)
        
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.classifier.eval()
        
        logger.info(f"Classifier loaded from epoch {checkpoint['epoch']}")
    
    def evaluate_video(self, video_path: str, verbose: bool = True) -> Dict:
        """
        Args:
            video_path: Path to the video file
            verbose: Whether to print detailed output
        
        Returns:
            Dictionary with evaluation results
        """

        #set base directory in the root so it can just be found by using data/processed/...
        base_dir = Path(__file__).parent.parent.parent
        video_path = os.path.join(base_dir, video_path)

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return {'error': 'Video file not found'}
        
        logger.info(f"\nEvaluating video: {video_path.name}")
        logger.info("=" * 60)
        
        frames, temporal_ids = encode_video(
            str(video_path),
            choose_fps=TARGET_FPS,
            target_duration=VIDEO_DURATION
        )
        
        if not frames:
            return {'error': 'Failed to encode video'}
        
        results = {}
        
        question = "Describe what is happening in this classroom video. Focus on the teaching and learning activities."
        msgs = [
            {'role': 'user', 'content': frames + [question]}
        ]
        
        try:
            with torch.no_grad():
                answer = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    use_image_id=False,
                    max_slice_nums=1,
                    temporal_ids=temporal_ids
                )
            
            results['description'] = answer
            
            if verbose:
                logger.info("\n--- Model Description ---")
                logger.info(answer)
            
            if self.classifier is not None:
                results['has_classifier'] = True
                
                # Note: This is simplified -need to do real implement (maybe if necessary) 
                with torch.no_grad():
                    # placeholder for actual feature extraction
                    features = torch.randn(1, 4096).to(self.device)
                    logits = self.classifier(features)
                    probs = torch.softmax(logits, dim=1)

                    #top preds 
                    top_k = 3
                    top_probs, top_indices = torch.topk(probs[0], top_k)
                    
                    predictions = []
                    for prob, idx in zip(top_probs, top_indices):
                        action_name = ACTIONS_REVERSE[idx.item()]
                        predictions.append({
                            'action': action_name,
                            'confidence': prob.item(),
                            'readable': action_name.replace('_', ' ').title()
                        })
                    
                    results['predictions'] = predictions
                    results['top_prediction'] = predictions[0]
                    
                    if verbose:
                        logger.info("\n--- Action Classification ---")
                        logger.info(f"Top Prediction: {predictions[0]['readable']} "
                                   f"(confidence: {predictions[0]['confidence']:.2%})")
                        logger.info("\nTop 3 Predictions:")
                        for i, pred in enumerate(predictions, 1):
                            logger.info(f"  {i}. {pred['readable']}: {pred['confidence']:.2%}")
            else:
                results['has_classifier'] = False

                # classify based on keywords in description 
                results['predictions'] = self.rule_based_classification(answer)
                
                if verbose and results['predictions']:
                    logger.info("\n--- Rule-Based Classification ---")
                    logger.info("(No fine-tuned classifier available, using keyword matching)")
                    logger.info(f"Detected Action: {results['predictions'][0]['readable']}")
        
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            results['error'] = str(e)
        
        results['video_path'] = str(video_path)
        results['timestamp'] = datetime.now().isoformat()
        results['model_type'] = 'fine-tuned' if self.classifier else 'base'
        
        if verbose:
            logger.info("\n" + "=" * 60)
        
        return results
    
    def rule_based_classification(self, description: str) -> List[Dict]:

        description_lower = description.lower()
        
        action_keywords = {
            'student_question': ['student asking', 'raising hand', 'student question', 'asks the'],
            'student_answer': ['student answering', 'student responds', 'student reply'],
            'student_discussion': ['students discussing', 'group discussion', 'peer discussion'],
            'student_presentation': ['student presenting', 'student at board', 'student demonstration'],
            'student_writing': ['students writing', 'taking notes', 'student writes'],
            'student_listening': ['students listening', 'students watching', 'paying attention'],
            'instructor_lecturing': ['instructor speaking', 'teacher explaining', 'professor talking', 'lecturing'],
            'instructor_writing': ['instructor writing', 'teacher at board', 'writing on board'],
            'instructor_demonstrating': ['instructor demonstrating', 'showing how', 'demonstration'],
            'instructor_question': ['instructor asking', 'teacher questions', 'asks students'],
            'instructor_answer': ['instructor answering', 'responding to student'],
            'instructor_moving': ['instructor walking', 'teacher moving', 'walking around'],
        }
        
        scores = {}
        for action, keywords in action_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                scores[action] = score
        
        if not scores:
            #default to most common action
            return [{
                'action': 'instructor_lecturing',
                'confidence': 0.1,
                'readable': 'Instructor Lecturing'
            }]
        
        total_score = sum(scores.values())
        predictions = []
        for action, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            predictions.append({
                'action': action,
                'confidence': score / total_score,
                'readable': action.replace('_', ' ').title()
            })
        
        return predictions[:3]  # Return top 3
    
    def evaluate_batch(self, video_dir: str, output_file: Optional[str] = None) -> List[Dict]:

        base_dir = Path(__file__).parent.parent.parent
        video_dir = os.path.join(base_dir, video_dir)
        video_dir = Path(video_dir)
        if not video_dir.exists():
            logger.error(f"Directory not found: {video_dir}")
            return []
        
        video_files = list(video_dir.glob("*.mp4"))
        if not video_files:
            logger.warning(f"No MP4 files found in {video_dir}")
            return []
        
        logger.info(f"\nEvaluating {len(video_files)} videos from {video_dir}")
        logger.info("=" * 60)
        
        results = []
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
            result = self.evaluate_video(str(video_file), verbose=False)
            results.append(result)
            
            if 'top_prediction' in result:
                logger.info(f"  -> {result['top_prediction']['readable']} "
                           f"({result['top_prediction']['confidence']:.1%})")
            elif 'predictions' in result and result['predictions']:
                logger.info(f"  -> {result['predictions'][0]['readable']} (rule-based)")
            elif 'error' in result:
                logger.info(f"  -> Error: {result['error']}")
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to: {output_path}")
        
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: List[Dict]):

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        total = len(results)
        successful = sum(1 for r in results if 'predictions' in r or 'description' in r)
        failed = sum(1 for r in results if 'error' in r)
        
        logger.info(f"Total videos: {total}")
        logger.info(f"Successfully evaluated: {successful}")
        logger.info(f"Failed: {failed}")
        
        if successful > 0:
            action_counts = {}
            for result in results:
                if 'predictions' in result and result['predictions']:
                    action = result['predictions'][0]['action']
                    action_counts[action] = action_counts.get(action, 0) + 1
                elif 'top_prediction' in result:
                    action = result['top_prediction']['action']
                    action_counts[action] = action_counts.get(action, 0) + 1
            
            if action_counts:
                logger.info("\nDetected Actions:")
                for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                    readable = action.replace('_', ' ').title()
                    percentage = (count / successful) * 100
                    logger.info(f"  - {readable}: {count} ({percentage:.1f}%)")
        
        logger.info("=" * 60)
    
    def compare_with_ground_truth(self, video_path: str, true_label: str) -> Dict:

        result = self.evaluate_video(video_path, verbose=True)
        
        if 'predictions' in result and result['predictions']:
            predicted_action = result['predictions'][0]['action']
            correct = predicted_action == true_label
            
            result['ground_truth'] = true_label
            result['correct'] = correct
            
            logger.info(f"\n--- Comparison with Ground Truth ---")
            logger.info(f"Ground Truth: {true_label.replace('_', ' ').title()}")
            logger.info(f"Prediction: {predicted_action.replace('_', ' ').title()}")
            logger.info(f"Result: {'✓ Correct' if correct else '✗ Incorrect'}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate COPUS video clips')
    parser.add_argument('video_path', type=str, help='Path to video file or directory')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--batch', action='store_true', help='Evaluate all videos in directory')
    parser.add_argument('--output', type=str, help='Output JSON file for batch results')
    parser.add_argument('--ground-truth', type=str, help='Ground truth label for comparison')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    evaluator = COPUSEvaluator(checkpoint_path=args.checkpoint)
    
    if args.batch:
        results = evaluator.evaluate_batch(args.video_path, args.output)
    elif args.ground_truth:
        result = evaluator.compare_with_ground_truth(args.video_path, args.ground_truth)
    else:
        result = evaluator.evaluate_video(args.video_path, verbose=True)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"\nResult saved to: {output_path}")
    


if __name__ == "__main__":

    #! this is for if we arent running in terminal and we arent passing any args (normally run it in terminal with specifications for batch etc.)
    if len(sys.argv) == 1:
        logger.info("Running in test mode with sample video...")
        
        evaluator = COPUSEvaluator()
        
        test_video = "data/processed/20241201/20241201_001.mp4"
        
        if Path(test_video).exists():
            result = evaluator.evaluate_video(test_video, verbose=True)
            
            print("\n" + "=" * 60)
            print("EVALUATION RESULT")
            print("=" * 60)
            print(json.dumps(result, indent=2))
        else:
            logger.warning(f"Test video not found: {test_video}")
    else:
        main()