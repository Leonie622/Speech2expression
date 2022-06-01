# Option 1:pix2pix approach，abandoned use
# Option 1:pix2pix approach，abandoned use
# Option 1:pix2pix approach，abandoned use
# Option 1:pix2pix approach，abandoned use
import warnings
warnings.filterwarnings('ignore')
from glob import glob

import cv2
import os

from speech_emotion.inference import predict_emotion
from utils import e_signs
from draw_sign_image import draw_sign


audio_path = "audios/angry/001.mp4"
# Audio Recognition Module
ret_emotion = predict_emotion(audio_path)
# Drawing signs
base_image_path = "videos/neutral/002/00000.jpg"
shape_name = e_signs[ret_emotion]
sign_image = draw_sign(cv2.imread(base_image_path),shape_name)
cv2.imwrite("temp_test/sign.jpg",sign_image)
# Implementing pix2pix model 
os.system("python test.py --label_nc 0 --no_instance --name temp_test  --dataroot ./temp_test/")


