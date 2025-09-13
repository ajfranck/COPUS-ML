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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoProcessor:

    def __init__(self, 
                raw_video_dir: str = "data/raw/videos",
                processed_dir: str = "data/processed",
                splice_duration: int = 120,
                buffer_duration: int = 10):