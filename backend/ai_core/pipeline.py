import os
import torch
import cv2
from PIL import Image
from torchvision.transform import transforms

class BrainTumorPipeline:

    def __init__(self, weights_dir: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
