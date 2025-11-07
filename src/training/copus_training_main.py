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
    """Production trainer with classifier head"""
    
    def __init__(
        self,
        base_model: str = 'openbmb/MiniCPM-V-4_5',
        output_dir: str = 'models/copus_production',
        device: str = 'cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo_id = "ajfranck/COPUS-analysis"  # Hardcoded - uploads automatically
        
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
        
        logger.info("✓ Vision-language model loaded and frozen")
        
        hidden_size = getattr(self.vl_model.config, 'hidden_size', 4096)
        self.classifier = COPUSClassifier(
            input_dim=hidden_size,
            num_classes=len(COPUS_ACTIONS)
        ).to(self.device)
        
        logger.info(f"✓ Classifier initialized (input: {hidden_size}, output: {len(COPUS_ACTIONS)})")
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def extract_features(self, frames: List[Image.Image], temporal_ids: List) -> torch.Tensor:
        """Extract features from frozen vision-language model"""
        with torch.no_grad():
            try:
                # Format data according to MiniCPM-V official API
                msgs = [{"role": "user", "content": frames + ["Describe the classroom activity."]}]
                
                if hasattr(self.vl_model, 'resampler') and hasattr(self.vl_model, 'vpm'):
                    try:
                        # Process images through vision encoder
                        from torchvision import transforms
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                        ])
                        
                        # Convert frames to tensor
                        frame_tensors = torch.stack([transform(f) for f in frames]).to(self.device)
                        
                        # Get vision features
                        vision_outputs = self.vl_model.vpm(frame_tensors)
                        
                        if hasattr(vision_outputs, 'last_hidden_state'):
                            vision_features = vision_outputs.last_hidden_state
                        else:
                            vision_features = vision_outputs
                        
                        if hasattr(self.vl_model, 'resampler'):
                            if len(vision_features.shape) == 4:  # [batch, channels, h, w]
                                b, c, h, w = vision_features.shape
                                vision_features = vision_features.flatten(2).transpose(1, 2)
                            
                            resampled = self.vl_model.resampler(vision_features)
                            features = resampled.mean(dim=1)  # [batch, hidden_dim]
                        else:
                            # No resampler, just pool the vision features
                            if len(vision_features.shape) == 4:
                                features = vision_features.mean(dim=[2, 3])  # Pool spatial dims
                            else:
                                features = vision_features.mean(dim=1)  # Pool sequence dim
                        
                        features = features.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                        return features
                        
                    except Exception as e:
                        logger.warning(f"Direct vision encoding failed: {e}, trying full model approach")
                
                #  Use full model forward pass with hidden states
                outputs = self.vl_model(
                    data=msgs,
                    tokenizer=self.tokenizer,
                    use_cache=False,
                    output_hidden_states=True
                )
                
                if hasattr(self.vl_model, 'resampler') and hasattr(outputs, 'vision_hidden_states'):
                    vision_features = outputs.vision_hidden_states[-1]  # Last layer
                    # Pool over sequence dimension
                    features = vision_features.mean(dim=1)  # [batch, hidden_dim]
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    # Use the last hidden state
                    last_hidden = outputs.hidden_states[-1]
                    # Pool over sequence dimension to get [1, hidden_dim]
                    features = last_hidden.mean(dim=1)
                else:
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        # Pool over sequence and vocab dimensions
                        features = logits.mean(dim=[1, 2]) if len(logits.shape) > 2 else logits.mean(dim=1)
                        # Project to expected dimension if needed
                        if features.shape[-1] != 4096:
                            features = torch.nn.functional.adaptive_avg_pool1d(
                                features.unsqueeze(1), 4096
                            ).squeeze(1)
                    else:
                        # Last resort we are cooked chat
                        logger.warning("Could not extract features, using random embedding")
                        features = torch.randn(1, 4096).to(self.device)
                
                return features
                
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                import traceback
                traceback.print_exc()
                return torch.randn(1, 4096).to(self.device)
    
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
                        
                        logits = self.classifier(features)
                        
                        loss = self.criterion(logits.squeeze(), labels)
                        loss.backward()
                        
                        predictions = torch.sigmoid(logits.squeeze()) > 0.5
                        correct = (predictions == labels).sum().item()
                        total = labels.numel()
                        
                        batch_loss += loss.item()
                        batch_correct += correct
                        batch_total += total
                    
                    except Exception as e:
                        logger.error(f"Error processing sample: {e}")
                        continue
                
                if batch_total > 0:
                    optimizer.step()
                    train_loss += batch_loss
                    train_correct += batch_correct
                    train_total += batch_total
                    
                    progress.set_postfix({
                        'loss': f'{batch_loss / len(batch):.4f}',
                        'acc': f'{100 * batch_correct / batch_total:.1f}%'
                    })
            
            avg_train_loss = train_loss / max(len(train_loader), 1)
            train_acc = 100 * train_correct / max(train_total, 1)
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            
            if val_dataset:
                val_loss, val_acc = self.validate(val_dataset)
                logger.info(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"✓ New best validation loss!")
                    self.save_checkpoint(f"best_model", epoch + 1, val_loss)
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}", epoch + 1, avg_train_loss)
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        
        final_name = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_checkpoint(final_name, epochs, avg_train_loss)
        self.upload_to_hub(final_name)
    
    def validate(self, val_dataset: COPUSDataset):
        """Validation"""
        self.classifier.eval()
        
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=lambda x: x
        )
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                for sample in batch:
                    frames = sample['frames']
                    temporal_ids = sample['temporal_ids']
                    labels = sample['labels'].to(self.device)
                    
                    if not frames:
                        continue
                    
                    try:
                        features = self.extract_features(frames, temporal_ids)
                        logits = self.classifier(features)
                        loss = self.criterion(logits.squeeze(), labels)
                        
                        predictions = torch.sigmoid(logits.squeeze()) > 0.5
                        correct = (predictions == labels).sum().item()
                        
                        total_loss += loss.item()
                        total_correct += correct
                        total_samples += labels.numel()
                    
                    except Exception as e:
                        logger.error(f"Validation error: {e}")
                        continue
        
        self.classifier.train()
        
        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = 100 * total_correct / max(total_samples, 1)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, name: str, epoch: int, loss: float):
        """Save checkpoint"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint: {checkpoint_dir}")
        
        # Save classifier
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.classifier.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_dir / 'classifier.pt')
        
        # Save base model info
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump({
                'base_model': 'openbmb/MiniCPM-V-4_5',
                'num_classes': len(COPUS_ACTIONS),
                'epoch': epoch,
                'loss': loss
            }, f, indent=2)
        
        logger.info("Checkpoint saved")
    
    def upload_to_hub(self, checkpoint_name: str):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("No HF_TOKEN environment variable found. Skipping upload.")
            return
        
        try:
            checkpoint_dir = self.output_dir / checkpoint_name
            
            if not checkpoint_dir.exists():
                logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
                return
            
            readme = f"""# COPUS Classifier

Classifier head for COPUS action recognition, trained on MiniCPM-V-4_5 features.

## Usage

```python
# Load classifier
classifier = torch.load('classifier.pt')

# Extract features from video
features = extract_features(frames, temporal_ids)

# Predict
logits = classifier(features)
predictions = torch.sigmoid(logits) > 0.5
```

## Actions: {len(COPUS_ACTIONS)}

Trained on: {datetime.now().strftime('%Y-%m-%d')}
"""
            with open(checkpoint_dir / 'README.md', 'w') as f:
                f.write(readme)
            
            logger.info(f"Uploading to HuggingFace: {self.hf_repo_id}")
            
            api = HfApi(token=hf_token)
            api.upload_folder(
                folder_path=str(checkpoint_dir),
                repo_id=self.hf_repo_id,
                repo_type="model",
            )
            
            logger.info(f"Successfully uploaded to https://huggingface.co/{self.hf_repo_id}")
        
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            logger.info("Model saved locally but not uploaded to HuggingFace")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train COPUS classifier (production)")
    parser.add_argument('--train-dir', required=True, help='Training videos directory')
    parser.add_argument('--train-labels', required=True, help='Training labels JSON')
    parser.add_argument('--val-dir', help='Validation videos directory')
    parser.add_argument('--val-labels', help='Validation labels JSON')
    parser.add_argument('--output-dir', default='models/copus_production')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    if not Path(args.train_dir).exists():
        logger.error(f"Training dir not found: {args.train_dir}")
        sys.exit(1)
    
    if not Path(args.train_labels).exists():
        logger.error(f"Labels file not found: {args.train_labels}")
        sys.exit(1)
    
    logger.info("Loading datasets...")
    train_dataset = COPUSDataset(args.train_dir, args.train_labels)
    
    val_dataset = None
    if args.val_dir and args.val_labels:
        if Path(args.val_dir).exists() and Path(args.val_labels).exists():
            val_dataset = COPUSDataset(args.val_dir, args.val_labels)
    
    trainer = COPUSTrainer(
        output_dir=args.output_dir,
        device=args.device
    )
    
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    logger.info(f"Output: {args.output_dir}")

if __name__ == "__main__":
    main()

# python copus_training_main.py --train-dir ../../data/processed/training --train-labels ../../data/processed/training/training_labels.json --epochs 10