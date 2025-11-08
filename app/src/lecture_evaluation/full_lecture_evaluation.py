import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
import math
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_NUM_FRAMES = 15
MAX_NUM_PACKING = 2
TIME_SCALE = 0.1
TARGET_FPS = 3
VIDEO_DURATION = 10.0

COPUS_ACTIONS = {
    "student_listening": 0, "student_individual_thinking": 1, "student_clicker_group": 2,
    "student_worksheet_group": 3, "student_other_group": 4, "student_answer_question": 5,
    "student_ask_question": 6, "student_whole_class_discussion": 7, "student_prediction": 8,
    "student_presentation": 9, "student_test_quiz": 10, "student_waiting": 11,
    "student_other": 12, "instructor_lecturing": 13, "instructor_real_time_writing": 14,
    "instructor_follow_up": 15, "instructor_posing_question": 16, "instructor_clicker_question": 17,
    "instructor_answering_question": 18, "instructor_moving_guiding": 19, "instructor_one_on_one": 20,
    "instructor_demo_video": 21, "instructor_administration": 22, "instructor_waiting": 23,
}

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
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]


def encode_video(video_path, choose_fps=3, target_duration=10.0):
    """Encode video for MiniCPM-V model"""
    def uniform_sample(l, n):
        if n >= len(l):
            return l
        gap = len(l) / n
        return [l[int(i * gap + gap / 2)] for i in range(n)]
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        video_duration = len(vr) / fps
        
        target_frames = int(target_duration * choose_fps)
        if target_frames <= MAX_NUM_FRAMES:
            packing_nums = 1
            choose_frames = min(target_frames, len(vr))
        else:
            packing_nums = min(math.ceil(target_frames / MAX_NUM_FRAMES), MAX_NUM_PACKING)
            choose_frames = target_frames
        
        frame_idx = uniform_sample(list(range(len(vr))), choose_frames)
        frame_idx = np.array(frame_idx)
        
        frames = vr.get_batch(frame_idx).asnumpy()
        frame_idx_ts = frame_idx / fps
        scale = np.arange(0, video_duration, TIME_SCALE)
        frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
        frame_ts_id = frame_ts_id.astype(np.int32)
        
        frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
        frame_ts_id_group = group_array(frame_ts_id, packing_nums)
        
        return frames, frame_ts_id_group
    except Exception as e:
        logger.error(f"Error encoding {video_path}: {e}")
        return [], []


class COPUSDataset(Dataset):
    """Dataset for COPUS video classification"""
    
    def __init__(self, data_dir: str, labels_file: str):
        self.data_dir = Path(data_dir)
        
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        
        self.samples = []
        for video_name, action_labels in labels_data.items():
            video_path = self.data_dir / video_name
            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            # Convert to list and create binary label vector
            if isinstance(action_labels, str):
                action_labels = [action_labels]
            
            label_vector = torch.zeros(len(COPUS_ACTIONS))
            for action in action_labels:
                if action in COPUS_ACTIONS:
                    label_vector[COPUS_ACTIONS[action]] = 1.0
            
            self.samples.append({
                'video_path': str(video_path),
                'labels': label_vector,
                'action_names': action_labels
            })
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames, temporal_ids = encode_video(sample['video_path'], TARGET_FPS, VIDEO_DURATION)
        
        return {
            'frames': frames,
            'temporal_ids': temporal_ids,
            'labels': sample['labels'],
            'action_names': sample['action_names'],
            'video_path': sample['video_path']
        }


class COPUSClassifier(nn.Module):
    """Multi-label classifier for COPUS actions"""
    
    def __init__(self, input_dim: int = 4096, num_classes: int = 24):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class COPUSTrainer:
    """Production trainer using model's built-in vision processing"""
    
    def __init__(
        self,
        base_model: str = 'openbmb/MiniCPM-V-4_5',
        output_dir: str = 'models/copus_production',
        device: str = 'cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo_id = "ajfranck/COPUS-analysis"
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Loading base model: {base_model}")
        
        self.vl_model = AutoModel.from_pretrained(
            base_model,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.vl_model = self.vl_model.to(self.device).eval()
        
        for param in self.vl_model.parameters():
            param.requires_grad = False
        
        logger.info("  Vision-language model loaded and frozen")
        
        hidden_size = getattr(self.vl_model.config, 'hidden_size', 4096)
        self.classifier = COPUSClassifier(
            input_dim=hidden_size,
            num_classes=len(COPUS_ACTIONS)
        ).to(self.device)
        
        logger.info(f"  Classifier initialized (input: {hidden_size}, output: {len(COPUS_ACTIONS)})")
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def extract_features(self, frames: List[Image.Image], temporal_ids: List) -> torch.Tensor:
        """
        Extract features using model.chat() with the SAME prompt as evaluation
        Uses all frames to maintain consistency with evaluation pipeline
        """
        with torch.no_grad():
            try:
                if not frames:
                    logger.warning("No frames provided")
                    return torch.randn(1, 4096, dtype=torch.float32).to(self.device)
                
                # Build COPUS actions list for prompt (same as evaluation)
                copus_actions_list = "\n".join([
                    f"- {key}: {value}" 
                    for key, value in COPUS_LABELS.items()
                ])
                
                # Use EXACT same prompt as evaluation script
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
                
                # Use ALL frames (just like evaluation)
                logger.info(f"Processing {len(frames)} frames with classification prompt")
                
                msgs = [{"role": "user", "content": frames + [classification_prompt]}]
                
                # Call model with same settings as evaluation
                response = self.vl_model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=500,  # Longer response for detailed analysis
                    sampling=False,
                    use_image_id=False,
                    max_slice_nums=1,  # Keep slicing minimal to avoid OOM
                )
                
                logger.info(f"✓ Model processed {len(frames)} frames successfully")
                logger.info(f"  Response preview: {response[:150]}...")
                
                # Extract embeddings from the response
                # The model has processed all images to generate this response
                tokens = self.tokenizer(
                    response,
                    return_tensors='pt',
                    max_length=512,  # Longer for detailed response
                    truncation=True,
                    padding=True
                )
                
                input_ids = tokens['input_ids'].to(self.device)
                
                # Get embeddings from LLM's embedding layer
                if hasattr(self.vl_model, 'llm'):
                    embed_layer = self.vl_model.llm.get_input_embeddings()
                    embeddings = embed_layer(input_ids)  # [1, seq_len, hidden_dim]
                    
                    # Average pool over sequence dimension
                    features = embeddings.mean(dim=1)  # [1, hidden_dim]
                    
                    # Convert to float32 to match classifier dtype
                    features = features.float()
                    
                    logger.info(f"✓ Extracted features: {features.shape}")
                    return features
                else:
                    logger.warning("Could not access LLM embeddings")
                    return torch.randn(1, 4096, dtype=torch.float32).to(self.device)
            
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to random
                logger.warning("Using random embedding fallback")
                return torch.randn(1, 4096, dtype=torch.float32).to(self.device)
    
    def train(
        self,
        train_dataset: COPUSDataset,
        val_dataset: Optional[COPUSDataset] = None,
        epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        save_every: int = 2
    ):
        """Train classifier head"""
        logger.info("=" * 60)
        logger.info("Starting Training (Classifier Head Approach)")
        logger.info("=" * 60)
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: x
        )
        
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*60}\nEpoch {epoch + 1}/{epochs}\n{'='*60}")
            
            # Training
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress = tqdm(train_loader, desc=f"Training")
            for batch in progress:
                batch_loss = 0.0
                batch_correct = 0
                batch_total = 0
                
                optimizer.zero_grad()
                
                for sample in batch:
                    frames = sample['frames']
                    temporal_ids = sample['temporal_ids']
                    labels = sample['labels'].to(self.device)
                    
                    if not frames:
                        continue
                    
                    try:
                        features = self.extract_features(frames, temporal_ids)
                        
                        # Convert to float32 to match classifier dtype
                        features = features.float()
                        
                        logits = self.classifier(features)
                        
                        loss = self.criterion(logits.squeeze(), labels)
                        
                        loss.backward()
                        
                        batch_loss += loss.item()
                        
                        preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                        batch_correct += (preds == labels).float().sum().item()
                        batch_total += labels.numel()
                        
                    except Exception as e:
                        logger.error(f"Error in training step: {e}")
                        continue
                
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                optimizer.step()
                
                if batch_total > 0:
                    train_loss += batch_loss
                    train_correct += batch_correct
                    train_total += batch_total
                    
                    avg_loss = train_loss / max(1, progress.n)
                    avg_acc = (train_correct / train_total * 100) if train_total > 0 else 0
                    progress.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.1f}%'})
            
            epoch_loss = train_loss / max(1, len(train_loader))
            epoch_acc = train_correct / max(1, train_total) * 100
            
            logger.info(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            
            scheduler.step(epoch_loss)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_dir = self.output_dir / f'checkpoint_epoch_{epoch+1}'
                checkpoint_dir.mkdir(exist_ok=True)
                
                classifier_path = checkpoint_dir / 'classifier.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'classifier_state_dict': self.classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_acc
                }, classifier_path)
                
                logger.info(f"Checkpoint saved: {checkpoint_dir}")
                
                # Upload to HuggingFace
                try:
                    hf_token = os.getenv("HF_TOKEN")
                    if hf_token:
                        api = HfApi(token=hf_token)
                        api.upload_file(
                            path_or_fileobj=str(classifier_path),
                            path_in_repo="classifier.pt",
                            repo_id=self.hf_repo_id,
                            repo_type="model"
                        )
                        logger.info(f"✓ Uploaded to HuggingFace: {self.hf_repo_id}")
                except Exception as e:
                    logger.warning(f"Could not upload to HuggingFace: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--train-labels', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='models/copus_production')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    
    train_dataset = COPUSDataset(args.train_dir, args.train_labels)
    
    trainer = COPUSTrainer(output_dir=args.output_dir)
    
    trainer.train(
        train_dataset=train_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()

# python copus_training_main.py --train-dir ../../data/processed/training --train-labels ../../data/processed/training/training_labels.json --epochs 3 --batch-size 4