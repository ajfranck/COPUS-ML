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
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
import math
import argparse
from collections import defaultdict
import time
import tempfile
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logging.warning("ffmpeg-python not installed. MTS conversion will not be available.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    'student_listening': 0,
    'student_individual_thinking': 1,
    'student_clicker_group': 2,
    'student_worksheet_group': 3,
    'student_other_group': 4,
    'student_answer_question': 5,
    'student_ask_question': 6,
    'student_whole_class_discussion': 7,
    'student_prediction': 8,
    'student_presentation': 9,
    'student_test_quiz': 10,
    'student_waiting': 11,
    'student_other': 12,
    
    'instructor_lecturing': 13,
    'instructor_real_time_writing': 14,
    'instructor_follow_up': 15,
    'instructor_posing_question': 16,
    'instructor_clicker_question': 17,
    'instructor_answering_question': 18,
    'instructor_moving_guiding': 19,
    'instructor_one_on_one': 20,
    'instructor_demo_video': 21,
    'instructor_administration': 22,
    'instructor_waiting': 23,
}

ACTIONS_REVERSE = {v: k for k, v in COPUS_ACTIONS.items()}

COPUS_LABELS = {
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
    """Map values to nearest scale point"""
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    """Group array into chunks of specified size"""
    return [arr[i:i+size] for i in range(0, len(arr), size)]


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
        
        frames = [Image.fromarray(frame.astype('uint8')).convert('RGB') for frame in frames_array]
        
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
        base_model: str = 'openbmb/MiniCPM-V-4_5',
        device: str = 'cuda',
        batch_size: int = 1
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        logger.info(f"Initializing Full Lecture Evaluator on device: {self.device}")
        
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
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            logger.info(f"Converting {mts_path.name} to temporary MP4...")
            
            (
                ffmpeg
                .input(str(mts_path))
                .output(
                    str(temp_path),
                    vcodec='libx264',
                    preset='ultrafast',
                    crf=23,
                    acodec='aac'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.info(f"Successfully converted to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error converting MTS file: {e}")
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            return None
    
    def evaluate_window(self, frames: List[Image.Image], temporal_ids: List) -> Dict:
        """
        Evaluate a single window of frames
        
        Returns:
            Dictionary with detected actions and confidence scores
        """
        question =  """Analyze this classroom video and describe in detail what both the instructor and students are doing. 

        For the INSTRUCTOR, observe and describe:
        - Is the instructor lecturing, presenting content, or explaining concepts to the class?
        - Are they writing or drawing on the board, document camera, or projector in real-time?
        - Are they asking questions to students (clicker questions, general questions, or checking understanding)?
        - Are they answering student questions or providing feedback to the class?
        - Are they moving around the classroom, checking on groups, or guiding student work?
        - Are they having one-on-one extended discussions with individual students?
        - Are they conducting a demonstration, showing a video, or running a simulation?
        - Are they handling administrative tasks (homework, attendance, announcements)?
        - Are they waiting or not actively engaging with students when they could be?

        For the STUDENTS, observe and describe:
        - Are students listening to the instructor, taking notes, or watching a presentation?
        - Are students thinking individually or working alone on problems after instructor prompts?
        - Are students discussing clicker questions in small groups or pairs?
        - Are students working together on worksheets or assigned group activities?
        - Are students engaged in other types of group discussions or collaborative work?
        - Is a student answering a question posed by the instructor with the class listening?
        - Are students asking questions or raising their hands for clarification?
        - Are students engaged in whole class discussion, sharing opinions or explanations?
        - Are students making predictions about demonstrations or experiments?
        - Are students presenting their work to the class?
        - Are students taking a test, quiz, or assessment?
        - Are students waiting due to delays, technical issues, or instructor being occupied?

        Describe specifically what you observe happening, including details about:
        - The type of activity (individual, small group, or whole class)
        - Whether the instructor is at the board, at the front, or moving around
        - If students are actively participating or passively receiving information
        - Any tools being used (clickers, worksheets, computers, lab equipment)
        - The level of interaction between instructor and students
        - Whether students are working independently or collaboratively

        Be specific and detailed in your observations."""
        msgs = [{'role': 'user', 'content': frames + [question]}]
        
        try:
            with torch.no_grad():
                answer = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    use_image_id=False,
                    max_slice_nums=1,
                    temporal_ids=temporal_ids
                )
            
            # this was if we wanted to use a classifier but seems like rule-based is better
            if self.classifier is not None:
                # some random placeholder thing i made
                features = torch.randn(1, 4096).to(self.device)
                logits = self.classifier(features)
                probs = torch.softmax(logits, dim=1)
                
                top_k = 5  # top 5 preds
                top_probs, top_indices = torch.topk(probs[0], top_k)
                
                predictions = {}
                for prob, idx in zip(top_probs, top_indices):
                    action_name = ACTIONS_REVERSE[idx.item()]
                    predictions[action_name] = prob.item()
            else:
                predictions = self.rule_based_classification(answer)
            
            return {
                'description': answer,
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error evaluating window: {e}")
            return {'error': str(e), 'predictions': {}}
    
    def rule_based_classification(self, description: str) -> Dict[str, float]:
        """Rule-based classification based on keywords"""
        description_lower = description.lower()
        
        action_keywords = {
            
            'student_listening': [
                # Core listening activities
                'listening', 'listening to instructor', 'listening to lecture', 
                'listening to presentation', 'listening to explanation',
                'paying attention', 'watching instructor', 'watching lecture',
                'observing instructor', 'looking at instructor', 'focused on instructor',
                
                # Note-taking activities
                'taking notes', 'writing notes', 'note taking', 'writing in notebook',
                'typing notes', 'recording notes', 'copying from board',
                'writing down information', 'transcribing', 'documenting',
                
                # Passive engagement
                'sitting quietly', 'watching presentation', 'viewing slides',
                'looking at board', 'looking at screen', 'watching demonstration',
                'observing lecture', 'following along', 'tracking presentation',
                'passive listening', 'receiving information', 'absorbing content'
            ],
            
            'student_individual_thinking': [
                # Individual problem solving
                'thinking individually', 'working alone', 'individual work',
                'solving problem alone', 'working independently', 'solo work',
                'thinking about question', 'considering problem', 'pondering',
                'individual problem solving', 'working by themselves',
                
                # Explicit thinking tasks
                'thinking about clicker question', 'considering clicker question',
                'individual reflection', 'personal reflection', 'thinking quietly',
                'working on problem individually', 'solving individually',
                'individual calculation', 'working out solution alone',
                
                # Silent work indicators
                'head down working', 'focused individual work', 'concentrated work',
                'silent thinking', 'quiet contemplation', 'individual analysis',
                'working without discussion', 'independent thinking',
                'solving on own', 'figuring out alone'
            ],
            
            'student_clicker_group': [
                # Clicker-specific group work
                'discussing clicker question', 'clicker discussion', 
                'clicker group work', 'clicker collaboration',
                'talking about clicker', 'clicker question discussion',
                'group clicker response', 'clicker peer discussion',
                
                # Group discussion indicators
                'discussing in pairs', 'discussing in groups of two',
                'small group clicker discussion', 'peer discussion clicker',
                'comparing clicker answers', 'sharing clicker responses',
                'debating clicker answer', 'clicker answer discussion',
                
                # Collaborative clicker work
                'working together on clicker', 'collaborative clicker',
                'group deliberation clicker', 'clicker consensus building',
                'peer instruction clicker', 'clicker peer learning'
            ],
            
            'student_worksheet_group': [
                # Worksheet-specific activities
                'working on worksheet', 'worksheet activity', 'worksheet group',
                'group worksheet', 'collaborative worksheet', 'worksheet discussion',
                'completing worksheet together', 'worksheet collaboration',
                
                # Problem set work
                'working on problem set', 'problem set discussion',
                'group problem solving worksheet', 'worksheet problems',
                'handout group work', 'working on handout together',
                
                # Structured group activities
                'assigned worksheet activity', 'structured group work',
                'guided worksheet activity', 'worksheet exercise',
                'lab worksheet', 'activity sheet', 'working through exercises'
            ],
            
            'student_other_group': [
                # General group activities
                'group discussion', 'working in groups', 'group work',
                'collaborative work', 'team discussion', 'peer discussion',
                'talking in groups', 'group conversation', 'team work',
                
                # Responding to instructor
                'responding to instructor question in groups',
                'group response to question', 'discussing instructor question',
                'group brainstorming', 'collective problem solving',
                
                # Various group tasks
                'assigned group activity', 'group project work',
                'peer collaboration', 'student collaboration',
                'working with partner', 'partner discussion',
                'small group activity', 'breakout discussion',
                'think-pair-share', 'peer learning activity'
            ],
            
            'student_answer_question': [
                # Answering instructor questions
                'student answering', 'student responds', 'answering question',
                'student response', 'providing answer', 'giving answer',
                'responding to instructor', 'answering instructor question',
                
                # With class listening
                'answering with class listening', 'public answer',
                'answering to whole class', 'class-wide response',
                'student explanation to class', 'sharing answer with class',
                
                # Various response types
                'verbal response', 'student reply', 'offering answer',
                'student solution', 'presenting answer', 'voicing answer',
                'student feedback', 'responding aloud', 'speaking up'
            ],
            
            'student_ask_question': [
                # Question asking
                'student asking', 'raising hand', 'asking question',
                'student question', 'posing question', 'student inquiry',
                'requesting clarification', 'seeking explanation',
                
                # Various question types
                'asking for help', 'requesting assistance', 'student query',
                'clarification question', 'follow-up question',
                'student asks instructor', 'questioning instructor',
                
                # Question indicators
                'hand raised', 'hand up', 'student interruption',
                'student clarification', 'asking about', 'inquiring about',
                'wants to know', 'confused and asking', 'seeking understanding'
            ],
            
            'student_whole_class_discussion': [
                # Whole class engagement
                'whole class discussion', 'class discussion', 'class-wide discussion',
                'engaged in discussion', 'participating in discussion',
                'offering explanation', 'sharing opinion', 'providing judgment',
                
                # Active participation
                'contributing to discussion', 'active discussion',
                'facilitated discussion', 'instructor-facilitated discussion',
                'sharing ideas with class', 'expressing viewpoint',
                
                # Various discussion types
                'debate participation', 'class dialogue', 'group dialogue',
                'exchanging ideas', 'class conversation', 'academic discourse',
                'thoughtful discussion', 'substantive discussion',
                'back and forth discussion', 'interactive discussion'
            ],
            
            'student_prediction': [
                # Making predictions
                'making prediction', 'predicting outcome', 'prediction about demo',
                'predicting result', 'forecasting outcome', 'anticipating result',
                'guessing outcome', 'hypothesizing result',
                
                # Demo/experiment predictions
                'demo prediction', 'experiment prediction', 'predict demonstration',
                'predict experiment', 'anticipating demo outcome',
                'forecasting experiment result', 'hypothesis about demo',
                
                # Prediction activities
                'writing prediction', 'stating prediction', 'sharing prediction',
                'discussing prediction', 'prediction activity',
                'pre-demo prediction', 'pre-experiment guess'
            ],
            
            'student_presentation': [
                # Student presentations
                'student presentation', 'presenting', 'student presenting',
                'giving presentation', 'student talk', 'student speech',
                'presenting to class', 'student demonstration',
                
                # Various presentation types
                'project presentation', 'group presentation',
                'individual presentation', 'poster presentation',
                'presenting findings', 'presenting solution',
                'presenting work', 'showing work to class',
                
                # Presentation activities
                'at front of class', 'using projector', 'presenting slides',
                'explaining to class', 'teaching class', 'peer teaching',
                'student lecture', 'student explanation'
            ],
            
            'student_test_quiz': [
                # Test/quiz activities
                'taking test', 'taking quiz', 'test', 'quiz', 'exam',
                'assessment', 'evaluation', 'taking exam',
                
                # Various assessment types
                'midterm', 'final exam', 'pop quiz', 'scheduled quiz',
                'written test', 'online quiz', 'clicker quiz',
                'practice test', 'diagnostic test',
                
                # Test-taking behaviors
                'working on test', 'completing quiz', 'answering test questions',
                'silent test taking', 'individual assessment',
                'timed assessment', 'formal evaluation'
            ],
            
            'student_waiting': [
                # Waiting scenarios
                'waiting', 'students waiting', 'idle', 'not engaged',
                'waiting for instructor', 'instructor late',
                'waiting for class to start', 'waiting for activity',
                
                # Technical delays
                'AV problems', 'technical difficulties', 'computer issues',
                'projector problems', 'waiting for technology',
                'equipment setup', 'technical delay',
                
                # Instructor occupied
                'instructor busy', 'instructor occupied', 'instructor elsewhere',
                'instructor helping others', 'waiting for instructor attention',
                'on hold', 'paused activity', 'downtime', 'break time'
            ],
            
            'student_other': [
                # Catch-all behaviors
                'other activity', 'unspecified activity', 'miscellaneous',
                'non-categorized', 'unique behavior', 'special case',
                
                # Off-task behaviors
                'checking phone', 'on laptop', 'distracted', 'off-task',
                'side conversation', 'packing up', 'arriving late',
                'leaving early', 'bathroom break', 'getting materials'
            ],
            
            # ============= INSTRUCTOR ACTIONS =============
            
            'instructor_lecturing': [
                # Core lecturing
                'lecturing', 'instructor speaking', 'presenting content',
                'delivering lecture', 'teaching', 'explaining concept',
                'instructor presentation', 'formal lecture',
                
                # Content delivery
                'presenting material', 'covering content', 'introducing topic',
                'explaining theory', 'discussing concept', 'teaching content',
                'content delivery', 'information presentation',
                
                # Mathematical/problem solving
                'deriving equation', 'mathematical derivation', 'deriving formula',
                'solving problem', 'working through problem', 'problem solution',
                'showing solution', 'demonstrating method', 'explaining steps',
                
                # Various lecture types
                'slide presentation', 'powerpoint presentation', 'verbal explanation',
                'theoretical explanation', 'conceptual overview', 'topic introduction',
                'summarizing material', 'reviewing content', 'lecture delivery'
            ],
            
            'instructor_real_time_writing': [
                # Board writing
                'writing on board', 'board work', 'whiteboard writing',
                'blackboard writing', 'drawing on board', 'sketching on board',
                'writing equations', 'drawing diagrams',
                
                # Document camera/projector
                'document camera writing', 'doc cam writing', 'projector writing',
                'overhead writing', 'writing on transparency',
                'annotating slides', 'marking up document',
                
                # Real-time creation
                'real-time writing', 'live writing', 'simultaneous writing',
                'writing while explaining', 'drawing while talking',
                'creating diagram', 'sketching illustration',
                'working out problem on board', 'step-by-step writing'
            ],
            
            'instructor_follow_up': [
                # Follow-up activities
                'follow-up', 'feedback', 'reviewing answers', 'discussing results',
                'clicker follow-up', 'activity follow-up', 'post-activity discussion',
                
                # Feedback to class
                'providing feedback', 'giving feedback', 'class-wide feedback',
                'reviewing responses', 'discussing answers', 'explaining results',
                'clarifying misconceptions', 'addressing errors',
                
                # Review activities
                'going over answers', 'reviewing solutions', 'recap activity',
                'summarizing responses', 'synthesizing ideas',
                'wrapping up activity', 'debriefing activity'
            ],
            
            'instructor_posing_question': [
                # Asking questions
                'asking question', 'posing question', 'instructor asks',
                'questioning students', 'prompting students', 'soliciting response',
                
                # Non-rhetorical questions
                'non-rhetorical question', 'genuine question', 'open question',
                'discussion question', 'thought question', 'concept check',
                'comprehension question', 'understanding check',
                
                # Various question types
                'probing question', 'follow-up question', 'clarifying question',
                'leading question', 'guided question', 'socratic questioning',
                'challenging students', 'prompting thinking', 'eliciting response'
            ],
            
            'instructor_clicker_question': [
                # Clicker questions
                'clicker question', 'asking clicker question', 'posing clicker',
                'clicker poll', 'audience response', 'polling students',
                'electronic response', 'clicker activity',
                
                # Clicker process
                'launching clicker', 'displaying clicker question',
                'reading clicker question', 'explaining clicker',
                'clicker voting', 'collecting responses', 'clicker time',
                
                # Duration markers
                'during clicker question', 'clicker question period',
                'entire clicker time', 'clicker session', 'response system active'
            ],
            
            'instructor_answering_question': [
                # Answering student questions
                'answering question', 'responding to student', 'instructor response',
                'addressing question', 'providing answer', 'explaining to student',
                
                # With class listening
                'answering with class listening', 'public response',
                'class-wide answer', 'answering for all',
                'broadcast answer', 'general response',
                
                # Various response types
                'clarifying for student', 'elaborating on answer',
                'detailed explanation', 'thorough response',
                'patient explanation', 'addressing confusion'
            ],
            
            'instructor_moving_guiding': [
                # Movement through class
                'walking around', 'moving through class', 'circulating',
                'moving between groups', 'visiting groups', 'checking on students',
                'roaming classroom', 'mobile instruction',
                
                # Guiding activities
                'guiding student work', 'facilitating groups', 'monitoring progress',
                'supervising activity', 'overseeing work', 'checking progress',
                'providing guidance', 'offering assistance',
                
                # Active learning support
                'supporting group work', 'facilitating discussion',
                'coaching students', 'mentoring groups', 'scaffolding learning',
                'proximity teaching', 'peripatetic teaching'
            ],
            
            'instructor_one_on_one': [
                # One-on-one interactions
                'one-on-one', 'individual discussion', 'private conversation',
                'extended discussion', 'personal interaction', 'individual help',
                
                # Small group focus
                'focused on one student', 'helping individual', 'individual attention',
                'concentrated assistance', 'dedicated help', 'exclusive attention',
                
                # Extended interactions
                'lengthy explanation', 'detailed individual help',
                'not paying attention to class', 'absorbed in helping',
                'deep dive with student', 'intensive assistance',
                'prolonged individual discussion'
            ],
            
            'instructor_demo_video': [
                # Demonstrations
                'demonstration', 'demo', 'showing demonstration', 'conducting demo',
                'performing demonstration', 'live demonstration', 'hands-on demo',
                
                # Experiments
                'experiment', 'conducting experiment', 'showing experiment',
                'laboratory demonstration', 'scientific demonstration',
                
                # Media
                'showing video', 'playing video', 'video presentation',
                'animation', 'simulation', 'showing simulation',
                'multimedia presentation', 'visual demonstration',
                
                # Various demo types
                'physical demonstration', 'equipment demonstration',
                'software demonstration', 'model demonstration',
                'illustrative example', 'practical example'
            ],
            
            'instructor_administration': [
                # Administrative tasks
                'administration', 'admin tasks', 'housekeeping',
                'assigning homework', 'giving assignment', 'announcing homework',
                'returning tests', 'returning papers', 'handing back work',
                
                # Class management
                'taking attendance', 'checking attendance', 'roll call',
                'making announcements', 'schedule announcements',
                'logistical information', 'course logistics',
                
                # Various admin activities
                'collecting homework', 'distributing materials', 'passing out papers',
                'organizing class', 'administrative duties', 'paperwork',
                'grade discussion', 'syllabus review', 'course management'
            ],
            
            'instructor_waiting': [
                # Instructor waiting
                'instructor waiting', 'standing idle', 'not interacting',
                'passive observation', 'watching students work',
                'available but not engaged', 'opportunity missed',
                
                # Non-engagement
                'not helping students', 'not circulating', 'stationary',
                'desk-bound', 'front of room only', 'not monitoring',
                
                # Missed opportunities
                'could be helping', 'should be guiding', 'passive supervision',
                'uninvolved', 'detached', 'hands-off approach',
                'waiting for students', 'letting students work alone'
            ]
        } 
        
        scores = {}
        for action, keywords in action_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                scores[action] = score
        
        # normalize scores
        if scores:
            total = sum(scores.values())
            return {action: score/total for action, score in scores.items()}
        else:
            # default to most common as a fallback (maybe remove this?)
            return {'instructor_lecturing': 0.5}
    
    def evaluate_full_lecture(self, video_path: str, output_path: Optional[str] = None, convert_mts: bool = True) -> Dict:
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
            return {'error': 'Video file not found'}
        
        logger.info(f"\nEvaluating full lecture: {video_path.name}")
        logger.info("=" * 60)
        
        temp_video_path = None
        original_path = video_path
        
        if video_path.suffix.upper() == '.MTS' and convert_mts:
            logger.info("Detected .MTS file. Converting to temporary .MP4 for processing...")
            temp_video_path = self.convert_mts_to_mp4(video_path)
            if temp_video_path:
                video_path = temp_video_path
                logger.info(f"Using temporary file: {temp_video_path}")
            else:
                logger.warning("MTS conversion failed, attempting direct processing...")
        
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            duration = total_frames / fps
            
            logger.info(f"Video info:")
            logger.info(f"  - Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info(f"  - FPS: {fps:.1f}")
            logger.info(f"  - Total frames: {total_frames}")
            
            if abs(fps - LECTURE_FPS) > 0.5:
                logger.warning(f"Warning: Expected {LECTURE_FPS}fps, got {fps:.1f}fps")
            
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
                    logger.warning(f"Failed to encode window {window_idx+1}/{num_windows}")
                    continue
                
                logger.info(f"Processing window {window_idx+1}/{num_windows} "
                           f"[{window_start_sec:.0f}s - {window_end_sec:.0f}s]")
                
                result = self.evaluate_window(frames, temporal_ids)
                
                window_results.append({
                    'window_idx': window_idx,
                    'start_time': window_start_sec,
                    'end_time': window_end_sec,
                    'predictions': result.get('predictions', {})
                })
                
                if (window_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / (window_idx + 1)) * (num_windows - window_idx - 1)
                    logger.info(f"  Progress: {window_idx+1}/{num_windows} windows "
                               f"(ETA: {eta/60:.1f} min)")
            
            aggregated_results = self.aggregate_results(window_results, duration)
            
            output = {
                'video_path': str(original_path),
                'video_info': {
                    'duration_seconds': duration,
                    'duration_minutes': duration / 60,
                    'fps': fps,
                    'total_frames': total_frames
                },
                'processing_info': {
                    'window_duration': WINDOW_DURATION,
                    'window_step': WINDOW_STEP,
                    'total_windows': len(window_results),
                    'aggregation_interval': AGGREGATION_INTERVAL
                },
                'intervals': aggregated_results,
                'timestamp': datetime.now().isoformat()
            }
            
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(output, f, indent=2)
                logger.info(f"\nResults saved to: {output_file}")
            
            self.print_summary(output)
            
            if temp_video_path and temp_video_path.exists():
                try:
                    temp_video_path.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_video_path}")
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
                    
            return {'error': str(e)}
    
    def aggregate_results(self, window_results: List[Dict], total_duration: float) -> List[Dict]:
        """
        Aggregate window results into 2-minute intervals
        
        Returns:
            List of dictionaries, one per 2-minute interval
        """
        num_intervals = int(math.ceil(total_duration / AGGREGATION_INTERVAL))
        intervals = []
        
        for interval_idx in range(num_intervals):
            interval_start = interval_idx * AGGREGATION_INTERVAL
            interval_end = min((interval_idx + 1) * AGGREGATION_INTERVAL, total_duration)
            
            overlapping_windows = []
            for window in window_results:
                if window['start_time'] < interval_end and window['end_time'] > interval_start:
                    overlapping_windows.append(window)
            
            actions_detected = defaultdict(float)
            for window in overlapping_windows:
                for action, confidence in window['predictions'].items():
                    actions_detected[action] = max(actions_detected[action], confidence)
            
            actions_binary = {}
            confidence_threshold = 0.3
            
            for action in COPUS_ACTIONS.keys():
                actions_binary[action] = actions_detected.get(action, 0) >= confidence_threshold
            
            intervals.append({
                'interval_number': interval_idx + 1,
                'start_time': interval_start,
                'end_time': interval_end,
                'start_time_str': str(timedelta(seconds=int(interval_start))),
                'end_time_str': str(timedelta(seconds=int(interval_end))),
                'actions': actions_binary,
                'actions_with_confidence': dict(actions_detected),
                'num_windows': len(overlapping_windows)
            })
        
        return intervals
    
    def print_summary(self, results: Dict):
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        if 'error' in results:
            logger.error(f"Evaluation failed: {results['error']}")
            return
        
        video_info = results['video_info']
        logger.info(f"Video: {Path(results['video_path']).name}")
        logger.info(f"Duration: {video_info['duration_minutes']:.1f} minutes")
        logger.info(f"Number of 2-minute intervals: {len(results['intervals'])}")
        
        action_counts = defaultdict(int)
        for interval in results['intervals']:
            for action, present in interval['actions'].items():
                if present:
                    action_counts[action] += 1
        
        if action_counts:
            logger.info("\nActions detected across all intervals:")
            sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
            
            for action, count in sorted_actions[:10]:  # Top 10 actions
                percentage = (count / len(results['intervals'])) * 100
                readable = COPUS_LABELS.get(action, action)
                logger.info(f"  - {readable}: {count}/{len(results['intervals'])} "
                           f"intervals ({percentage:.1f}%)")
        
        logger.info("\nActivity Timeline (first 10 intervals):")
        for interval in results['intervals'][:10]:
            active_actions = [action for action, present in interval['actions'].items() if present]
            if active_actions:
                top_actions = active_actions[:3]
                action_str = ", ".join([COPUS_LABELS.get(a, a).split(' - ')[0] for a in top_actions])
                logger.info(f"  {interval['start_time_str']} - {interval['end_time_str']}: {action_str}")
        
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate full COPUS lecture video')
    parser.add_argument('video_path', type=str, help='Path to full lecture video (3fps, .mp4 or .mts)')
    parser.add_argument('--output', '-o', type=str, required=True, 
                       help='Output JSON file path for results')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=1, 
                       help='Batch size for processing')
    parser.add_argument('--no-convert-mts', action='store_true',
                       help='Attempt to process MTS files directly without conversion')
    
    args = parser.parse_args()
    
    evaluator = FullLectureEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size
    )
    
    results = evaluator.evaluate_full_lecture(
        args.video_path, 
        args.output,
        convert_mts=(not args.no_convert_mts)
    )
    
    if 'error' not in results:
        logger.info(f"\nEvaluation complete! Results saved to: {args.output}")
        
        logger.info("\nSample of interval data (first interval):")
        if results['intervals']:
            first_interval = results['intervals'][0]
            logger.info(f"  Time: {first_interval['start_time_str']} - {first_interval['end_time_str']}")
            logger.info("  Actions present:")
            for action, present in first_interval['actions'].items():
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
            Path("data/raw/videos/20191101/00003.mp4"),
        ]
        
        for test_video in test_videos:
            if test_video.exists():
                logger.info(f"\nFound test video: {test_video}")
                evaluator = FullLectureEvaluator()
                results = evaluator.evaluate_full_lecture(
                    str(test_video), 
                    "data/results/test_lecture_copus.json"
                )
                break
        else:
            logger.info("\nNo test video found. Please provide a video path.")
    else:
        main()