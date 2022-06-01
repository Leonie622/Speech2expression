import cv2
import numpy as np

# Define the signs, draw the signs on the image
def draw_sign(_image, _sign_name):
    _img_h, _img_w = _image.shape[:2]
    if _sign_name == "white_circular":
        _center = (_img_w // 2, _img_h // 2)
        cv2.circle(_image, _center, 60, (255, 255, 255), -1)
    elif _sign_name == "circular":
        _center = (_img_w // 2, _img_h // 2)
        cv2.circle(_image, _center, 60, (255, 255, 255), 2)
    elif _sign_name == "white_square":
        _center_w, _center_h = _img_w // 2, _img_h // 2
        _length = 50
        cv2.rectangle(_image, (_center_w - _length, _center_h - _length),
                      (_center_w + _length, _center_h + _length), (255, 255, 255), -1)
    elif _sign_name == "red_square":
        _center_w, _center_h = _img_w // 2, _img_h // 2
        _length = 60
        cv2.rectangle(_image, (_center_w - _length, _center_h - _length),
                      (_center_w + _length, _center_h + _length), (0, 0, 255), -1)
    elif _sign_name == "red_circular":
        _center = (_img_w // 2, _img_h // 2)
        cv2.circle(_image, _center, 60, (0, 0, 255), -1)
    return _image

if __name__ == "__main__":
    from utils import signs
    import time
    for sign in signs:
        image = cv2.imread("videos/neutral/001/00000.jpg")
        # image = np.zeros((300, 300, 3), dtype="uint8")
        ret_img = draw_sign(image,sign)
        cv2.imwrite("ret_{}.jpg".format(sign),ret_img)
        # time.sleep(1)
