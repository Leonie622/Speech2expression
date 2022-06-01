from glob import glob
import cv2
import os
from tqdm import tqdm

# Extract all frames of the video
def extract_all_frames(_video_path:str):
    cap = cv2.VideoCapture(_video_path)
    frame_path = _video_path.split(".")[0]
    os.makedirs(frame_path,exist_ok=True)
    count = 0
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frame_path,"{}.jpg".format(str(count).zfill(5))),frame)
        count += 1

# Extract all MP4 images from the frames
def extract_all_videos():
    root = r"videos/*/*.mp4"
    video_path_list = glob(root)
    for video_path in tqdm(video_path_list):
        extract_all_frames(video_path)


root = r"videos/*/*/*.jpg"
video_path_list = glob(root)
print(len(video_path_list))

