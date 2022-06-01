#conding=utf-8
#Set emotion weight parameters
e_lamdbda = {
    "angry":-100.,
    "netural":1.2,
    "fear":-1.,
    "sad":0.2,
    "happy":100.5
}

#from draw_sign_image import draw_sign
#from utils import e_lamdbda
from speech_emotion.inference import predict_emotion
import os
import cv2
from glob import glob
import warnings
warnings.filterwarnings('ignore')

### Input parameters
audio_path = "audios/angry/001.wav" # Path of audio
img_path = r"IALS-main/image/01.jpg"  # Path to face image, using an absolute path
out_img_path = "IALS-main/image_output/001.wav4b23.jpg"  # Path to generate and save face images


# Audio Recognition Module
ret_emotion = predict_emotion(audio_path)


# Option 1:pix2pix approach，abandoned use
# Drawing signs
# base_image_path = "videos/neutral/002/00000.jpg"
# shape_name = e_signs[ret_emotion]
# sign_image = draw_sign(cv2.imread(base_image_path),shape_name)
# cv2.imwrite("temp_test/sign.jpg",sign_image)
# Implementing pix2pix model 
# os.system("python test.py --label_nc 0 --no_instance --name temp_test  --dataroot ./temp_test/")

# Option 2
# Implementing the IALS project
cmd1 = "python gan_inversion.py --n_iters 100  --img_path {}".format(img_path)
cmd2 = "python condition_manipulation.py --seed 0 --step -0.1 --n_steps 10 --dataset ffhq --base interfacegan\
 --attr1 smiling --attr2 young --lambda1 {} --lambda2 0.3 --real_image 1 --latent_code_path rec.npy --save_path {}".format(e_lamdbda[ret_emotion], out_img_path)


os.chdir("IALS-main")
# os.system("cd IALS-main")
os.system(cmd1)
os.system(cmd2)
