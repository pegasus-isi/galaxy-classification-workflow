#!/usr/bin/env python3

import random
import numpy as np
import os
import glob
import cv2
import argparse
import time



#| Class    |  Train |  Val   | Test   |
#|----------|:------:|:------:|:------:|
#| class 0  |  6733  |  858   |  845   |
#| class 1  |  6449  |  817   |  803   |
#| class 2  |   464  |   62   |   53   |
#| class 3  |  3141  |  364   |  398   |
#| class 4  |  6247  |  778   |  781   |
#--------------------------------------|
#| TOTAL    | 23034  | 2879   | 2880   |


IMG_SIZE    = 150
DATASET_DIR = "./"


def get_arguments():
    
    parser = argparse.ArgumentParser(description="Galaxy Classification: data augmentation")   
    parser.add_argument('--class_str', type=str, default='class_2',help='class to augment')
    parser.add_argument('--num', type=int, default=10,help='number of images to create')  
    parser.add_argument(
        "-f","--input_files", nargs='*', default="",
        help="input data, overrides input dir"
        )
    parser.add_argument(
        "-i","--input_dir",default="",
        help="directory with data"
        )
    parser.add_argument(
        "-o","--output_dir",default="",
        help="directory for outputs"
        ) 
    args = parser.parse_args()    
    return args


class DataAugmentation:
   
    def load_img(self,img_path):
        return cv2.imread(img_path)

    def rotate(self, image, angle=20, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image

    
    def image_augment(self, path): 
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        '''
        p = np.random.uniform(0, 1)
        img = self.load_img(path)
        img = img/255

        if p < 0.51:
            img = self.rotate(img, 130, 0.80)
            #crop
            img = img[50:206,50:206,:]
            img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            # resize
        else:
            img = self.rotate(img,70, 1.15)
        return img

# 

def add_augmented_data(class_str, num):
    augmentation = DataAugmentation()
    
    if not INPUT_FILES:
        all_images = glob.glob(os.path.join(INPUT_DIR, "train_" + class_str + "*.jpg"))
    else:
        all_images = INPUT_FILES

    new_images = 0
    img_num = 4000
    while new_images < num:
        for img_path in all_images:
            augmented_img = augmentation.image_augment(img_path)
            path_s, _ = img_path.split(".")
            fname = "_".join(path_s.split("_")[:3])
            fname = fname  + "_" + str(img_num) + "_proc.jpg"
            cv2.imwrite(fname,augmented_img*255)
            new_images +=1
            img_num+=1
            if new_images >= num:
                break


def main():
    
    global ARGS
    global INPUT_DIR
    global INPUT_FILES
    global OUTPUT_DIR
    
    start      = time.time()
    ARGS       = get_arguments()   
    class_str  = ARGS.class_str
    num        = ARGS.num
    INPUT_DIR  = ARGS.input_dir
    INPUT_FILES = ARGS.input_files
    OUTPUT_DIR = ARGS.output_dir

    add_augmented_data(class_str, num)
    exec_time = time.time() - start
    print('Execution time in seconds: ' + str(exec_time))



if __name__ == "__main__":
    
    main()
