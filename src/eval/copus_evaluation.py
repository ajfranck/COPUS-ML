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

# Updated COPUS actions based on the new list
COPUS_ACTIONS = {
    # Student actions
    'student_listening': 0,  # L - Listening to instructor/taking notes
    'student_individual_thinking': 1,  # Ind - Individual thinking/problem solving
    'student_clicker_group': 2,  # CG - Discuss clicker question in groups
    'student_worksheet_group': 3,  # WG - Working in groups on worksheet
    'student_other_group': 4,  # OG - Other assigned group activity
    'student_answer_question': 5,  # AnQ - Answering instructor question
    'student_ask_question': 6,  # SQ - Student asks question
    'student_whole_class_discussion': 7,  # WC - Whole class discussion
    'student_prediction': 8,  # Prd - Making prediction
    'student_presentation': 9,  # SP - Presentation by student(s)
    'student_test_quiz': 10,  # TQ - Test or quiz
    'student_waiting': 11,  # W - Waiting
    'student_other': 12,  # O - Other
    
    # Instructor actions
    'instructor_lecturing': 13,  # Lec - Lecturing
    'instructor_real_time_writing': 14,  # RtW - Real-time writing
    'instructor_follow_up': 15,  # FUp - Follow-up/feedback on clicker
    'instructor_posing_question': 16,  # PQ - Posing non-clicker question
    'instructor_clicker_question': 17,  # CQ - Asking clicker question
    'instructor_answering_question': 18,  # AnQ - Answering student questions
    'instructor_moving_guiding': 19,  # MG - Moving through class guiding
    'instructor_one_on_one': 20,  # 1o1 - One-on-one discussion
    'instructor_demo_video': 21,  # D/V - Demo, experiment, video
    'instructor_administration': 22,  # Adm - Administration
    'instructor_waiting': 23,  # W - Waiting
}

ACTIONS_REVERSE = {v: k for k, v in COPUS_ACTIONS.items()}

# Human-readable labels with COPUS codes
COPUS_LABELS = {
    # Student actions
    'student_listening': 'L - Listening/Taking Notes',
    'student_individual_thinking': 'Ind - Individual Thinking',
    'student_clicker_group': 'CG - Clicker Group Discussion',
    'student_worksheet_group': 'WG - Worksheet Group Work',
    'student_other_group': 'OG - Other Group Activity',
    'student_answer_question': 'AnQ - Answering Question',
    'student_ask_question': 'SQ - Asking Question',
    'student_whole_class_discussion': 'WC - Whole Class Discussion',
    'student_prediction': 'Prd - Making Prediction',
    'student_presentation': 'SP - Student Presentation',
    'student_test_quiz': 'TQ - Test/Quiz',
    'student_waiting': 'W - Waiting',
    'student_other': 'O - Other',
    
    # Instructor actions
    'instructor_lecturing': 'Lec - Lecturing',
    'instructor_real_time_writing': 'RtW - Real-time Writing',
    'instructor_follow_up': 'FUp - Follow-up/Feedback',
    'instructor_posing_question': 'PQ - Posing Question',
    'instructor_clicker_question': 'CQ - Clicker Question',
    'instructor_answering_question': 'AnQ - Answering Questions',
    'instructor_moving_guiding': 'MG - Moving/Guiding',
    'instructor_one_on_one': '1o1 - One-on-One',
    'instructor_demo_video': 'D/V - Demo/Video',
    'instructor_administration': 'Adm - Administration',
    'instructor_waiting': 'W - Waiting',
}


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
                            'readable': COPUS_LABELS.get(action_name, action_name),
                            'code': action_name
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
        
        # Updated keyword mappings for new COPUS categories
        action_keywords = {
            # Student actions
            'student_listening': [
                'students listening', 'taking notes', 'students watching instructor',
                'paying attention', 'students focused on', 'listening to lecture',
                'watching the board', 'students observing'
            ],
            'student_individual_thinking': [
                'thinking individually', 'working alone', 'solving problems individually',
                'individual work', 'thinking on their own', 'working independently',
                'students thinking', 'reflecting individually'
            ],
            'student_clicker_group': [
                'clicker discussion', 'discussing clicker', 'clicker groups',
                'talking about clicker', 'clicker question discussion'
            ],
            'student_worksheet_group': [
                'worksheet groups', 'working on worksheet', 'group worksheet',
                'worksheet activity', 'completing worksheet together'
            ],
            'student_other_group': [
                'group discussion', 'students discussing', 'group work',
                'working in groups', 'collaborative work', 'peer discussion',
                'students talking', 'group activity'
            ],
            'student_answer_question': [
                'student answering', 'student responds', 'student reply',
                'answering question', 'student explains', 'student response'
            ],
            'student_ask_question': [
                'student asking', 'raising hand', 'student question',
                'asks the instructor', 'student inquires', 'asks about'
            ],
            'student_whole_class_discussion': [
                'class discussion', 'whole class', 'class debate',
                'everyone discussing', 'class conversation', 'sharing with class'
            ],
            'student_prediction': [
                'making prediction', 'predicting outcome', 'students predict',
                'guessing result', 'hypothesis', 'students anticipate'
            ],
            'student_presentation': [
                'student presenting', 'student at board', 'student demonstration',
                'presenting to class', 'student explains to class', 'student shows'
            ],
            'student_test_quiz': [
                'taking test', 'quiz', 'exam', 'assessment', 'students testing',
                'completing quiz', 'test in progress'
            ],
            'student_waiting': [
                'students waiting', 'idle', 'not engaged', 'waiting for',
                'students inactive', 'no activity'
            ],
            
            # Instructor actions
            'instructor_lecturing': [
                'instructor speaking', 'teacher explaining', 'professor talking',
                'lecturing', 'presenting content', 'instructor teaches',
                'delivering lecture', 'explaining concepts'
            ],
            'instructor_real_time_writing': [
                'writing on board', 'instructor writing', 'drawing on board',
                'writing equations', 'board work', 'whiteboard writing',
                'projector writing', 'annotating'
            ],
            'instructor_follow_up': [
                'feedback on question', 'reviewing answers', 'discussing results',
                'follow-up', 'explaining answers', 'clarifying responses',
                'reviewing clicker', 'going over answers'
            ],
            'instructor_posing_question': [
                'asking question', 'instructor asks', 'posing question',
                'questioning students', 'teacher questions', 'prompting students'
            ],
            'instructor_clicker_question': [
                'clicker question', 'using clicker', 'polling students',
                'clicker poll', 'electronic response', 'clicker activity'
            ],
            'instructor_answering_question': [
                'answering student', 'responding to question', 'instructor responds',
                'addressing question', 'teacher answers', 'replying to student'
            ],
            'instructor_moving_guiding': [
                'walking around', 'moving through class', 'circulating',
                'checking on groups', 'monitoring students', 'guiding groups',
                'instructor moving', 'walking between'
            ],
            'instructor_one_on_one': [
                'individual help', 'one-on-one', 'helping individual',
                'personal assistance', 'individual discussion', 'private conversation'
            ],
            'instructor_demo_video': [
                'demonstration', 'showing video', 'experiment', 'demo',
                'simulation', 'showing animation', 'displaying video',
                'conducting experiment', 'visual demonstration'
            ],
            'instructor_administration': [
                'taking attendance', 'handing out', 'collecting papers',
                'administrative', 'homework assignment', 'returning tests',
                'class logistics', 'announcements'
            ],
            'instructor_waiting': [
                'instructor waiting', 'teacher idle', 'not teaching',
                'instructor paused', 'waiting for students'
            ],
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
                'readable': COPUS_LABELS['instructor_lecturing'],
                'code': 'instructor_lecturing'
            }]
        
        total_score = sum(scores.values())
        predictions = []
        for action, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            predictions.append({
                'action': action,
                'confidence': score / total_score,
                'readable': COPUS_LABELS.get(action, action),
                'code': action
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
                    readable = COPUS_LABELS.get(action, action)
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
            logger.info(f"Ground Truth: {COPUS_LABELS.get(true_label, true_label)}")
            logger.info(f"Prediction: {COPUS_LABELS.get(predicted_action, predicted_action)}")
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