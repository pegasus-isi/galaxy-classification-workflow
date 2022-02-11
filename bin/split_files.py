import os
import urllib.request
import shutil
import numpy as np
import zipfile
import random
import glob
import sys
import argparse
from os import path







def parse_args(args):
    parser = argparse.ArgumentParser(description="Enter description here")
    parser.add_argument(
        "-i","--input_dir",default=".",
        help="directory with data"
        )
    parser.add_argument(
        "-o","--output_dir",default=".",
        help="directory for outputs"
        )
    parser.add_argument(
        "-s","--seed",type= int,default=10,
        help="seed for random"
        )
    return parser.parse_args(args)




def add_prefix(file_paths, prefix, output_dir):
    new_paths = []
    for fpath in file_paths:
        path, fname = fpath.split('/')
        fname = prefix + "_" + fname
        os.rename(fpath,output_dir+fname)
    return new_paths


def split_data_filenames(file_paths,seed):
    random.seed(seed)
    random.shuffle(file_paths)
    train, val, test = np.split(file_paths, [int(len(file_paths)*0.8), int(len(file_paths)*0.9)])
    return train, val, test



def main():
    
    args = parse_args(sys.argv[1:])
    input_dir  = args.input_dir
    output_dir = args.output_dir
    seed       = args.seed
    
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    all_images       = glob.glob(input_dir + "*.jpg")
    train, val, test = split_data_filenames(all_images,seed)
    pf_train = add_prefix(train, "train",output_dir)
    pf_val   = add_prefix(val, "val",output_dir)
    pf_test  = add_prefix(test, "test",output_dir)





if __name__ == '__main__':
	main()


