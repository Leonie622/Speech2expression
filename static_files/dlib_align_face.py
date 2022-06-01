import argparse
import numpy as np
import cv2
import dlib
from skimage import transform as trans

parser = argparse.ArgumentParser(description="put image need to align...")                                                                            
 # These parameters have default values, and when calling parser.print_help() or running the program with incorrect parameters (when the python interpreter actually calls the pring_help() method as well)
# This description information will be printed, and generally only the description parameter needs to be passed, as above.
parser.add_argument('--path',default="", help = "The path of image")
args = parser.parse_args() 

def shape_to_np(shape, dtype="int"):
# Convert a shape containing 68 features to numpy array format
coords = np.zeros((5, 2), dtype=dtype)
    for i in range(0, 5):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        print("x,y:",shape.part(i).x,shape.part(i).y)
    return coords

#Adapt from https://www.programcreek.com/python/example/103115/dlib.shape_predictor
# dlib Detecting face feature point
def detect_landmark_dlib(_cv2image):
    _gray = cv2.cvtColor(_cv2image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("static_files/shape_predictor_5_face_landmarks.dat")
    _rects = detector(_gray, 1)
    shapes = []
    for (i, rect) in enumerate(_rects):
        shape = predictor(_gray, rect)
        shape = shape_to_np(shape)
        shapes.append(shape)
    return np.array(shapes,dtype=np.float32).squeeze(0)

# Generic face alignment method
# "rate" determines the face scaling ratio,
# "align_out_size" determines the output image size.

def get_align(cv2img, npy_path="static_files/FFHQ_template.npy", align_out_size=(512,512), rate=1):
    landmark_points = detect_landmark_dlib(cv2img)
    print("landmark shape:",landmark_points.shape)
    ref_points = np.load(npy_path)
    print("ref_points shape:",ref_points.shape)
    align_center = np.array([align_out_size[1] // 2, align_out_size[0] // 2])
    tform = trans.SimilarityTransform()
    size_rate = (align_out_size[1] / 1024, align_out_size[0] / 1024)
    ref_points[..., 0] *= size_rate[0]
    ref_points[..., 1] *= size_rate[1]
    ref_points = ref_points.astype(np.float32)
    off_points = ref_points - align_center
    ref_points = align_center + rate * off_points
    tform.estimate(landmark_points, ref_points)
    result = trans.warp(cv2img, tform.inverse, output_shape=align_out_size, order=3)
    return result

if __name__ == "__main__":
    image = cv2.imread(args.path)
    # image = cv2.resize(image,(512,512))
    print("image shape:",image.shape)
    res = (get_align(image)*255).astype(np.uint8)
    cv2.imwrite("01030.jpg ",res)
