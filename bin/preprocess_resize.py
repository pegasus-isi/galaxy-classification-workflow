#!/usr/bin/env python3
import random
import glob
import sys
import argparse
import os
from PIL import Image



def parse_args(args):
    parser = argparse.ArgumentParser(description="Enter description here")
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
    return parser.parse_args(args)



def main():
    args = parse_args(sys.argv[1:])
    
    input_files = args.input_files
    if not input_files:
        input_dir  = args.input_dir
        all_images = glob.glob(os.path.join(input_dir,"*.jpg"))
    else:
        all_images = input_files
    
    for img_path in all_images:
        img = Image.open(img_path)
        width, height = img.size
        new_width = 256
        new_height = 256

        left   = (width - new_width)/2
        top    = (height - new_height)/2
        right  = (width + new_width)/2
        bottom = (height + new_height)/2

        img = img.crop((left, top, right, bottom))
        img_path = img_path.split(".")[0]
        img_path = img_path + "_proc.jpg"
        img.save(img_path)




if __name__ == '__main__':
	main()


