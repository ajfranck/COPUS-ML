import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
import math
import random