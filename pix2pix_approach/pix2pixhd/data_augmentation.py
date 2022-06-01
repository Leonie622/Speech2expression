
##Adapt from https://www.tensorflow.org/tutorials/images/data_augmentation
##Data augmentation of input data##

import os
import cv2


def data_augmentation(input_path, output_path):
    """a method to augmentate dataset

    """
    for fn in os.listdir(input_path):
        try:
            pic_name = os.path.join(input_path, fn)
            pic = cv2.imread(pic_name)
            h_filp = cv2.flip(pic, 1)  # Horizontal Mirror
            v_filp = cv2.flip(pic, 0)  # Vertical Mirror
            hv_filp = cv2.flip(pic, -1)  # Horizontal and vertical mirror
            rows, cols = pic.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
            # Rotated 90 degrees, border filled with white
            dst_90 = cv2.warpAffine(pic, M, (cols, rows), borderValue=(255, 255, 255))  

            # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
            # Rotated 45 degrees, border filled with white
            # dst_45 = cv2.warpAffine(pic, M, (cols, rows), borderValue=(255, 255, 255))  
            cv2.imwrite(output_path + 'h_filp_' + fn, h_filp)
            cv2.imwrite(output_path + 'v_filp_' + fn, v_filp)
            cv2.imwrite(output_path + 'hv_filp_' + fn, hv_filp)
            cv2.imwrite(output_path + 'dst_90_' + fn, dst_90)
            # cv2.imwrite(output_path + 'dst_45_' + fn, dst_45)
        except:
            print(pic_name)


if __name__=='__main__':

    input_path = "./datasets/befunky_3/train_B/"
    output_path = "./datasets/befunky_3/train_B/"

    data_augmentation(input_path, output_path)

     
