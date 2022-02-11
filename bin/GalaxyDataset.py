#!/usr/bin/env python3
import numpy as np
import pandas as pd
from argparse import ArgumentParser

MAX_IMG_DEFAULT = [100, 100, 100, 100, 100]

class GalaxyDataset():
    def __init__(self, max_img=MAX_IMG_DEFAULT, seed=10, metadata="training_solutions_rev1.csv", available_images_log="full_galaxy_data.log"):
        if len(max_img) < 5:
            self.max_img = MAX_IMG_DEFAULT
        elif len(max_img) > 5:
            self.max_img = max_img[:5]
        else:
            self.max_img = max_img
        self.seed = seed
        self.metadata = metadata
        self.available_images_log = available_images_log
        self.labeled_dataset = None
        self.available_images = None
        return

    def read_available_images(self):
        with open(self.available_images_log, 'r') as f:
            self.available_images = f.read().splitlines()
        return

    def label_dataset(self):        
        final_table_columns = ['GalaxyID', 'Class1.1', 'Class1.2', 'Class2.1', 'Class2.2', 'Class4.1', 'Class7.1', 'Class7.2', 'Class7.3']
    
        df = pd.read_csv(self.metadata)
        df.drop(columns = df.columns.difference(final_table_columns), inplace=True)
    
        #create label column with default value
        df['Label'] = '$'
    
        df['Label'] = np.where((df['Class1.1'] >= 0.469) & (df['Class7.1'] >= 0.500), '0', df['Label'])
        df['Label'] = np.where((df['Label'] == '$') & (df['Class1.1'] >= 0.469) & (df['Class7.2'] >= 0.500), '1', df['Label'])
        df['Label'] = np.where((df['Label'] == '$') & (df['Class1.1'] >= 0.469) & (df['Class7.3'] >= 0.500), '2', df['Label'])
        df['Label'] = np.where((df['Label'] == '$') & (df['Class1.2'] >= 0.430) & (df['Class2.1'] >= 0.602), '3', df['Label'])
        df['Label'] = np.where((df['Label'] == '$') & (df['Class1.2'] >= 0.430) & (df['Class2.2'] >= 0.715) & (df['Class4.1'] >= 0.619), '4', df['Label'])

        #drop rows where Label is still '$'
        df = df[df['Label'] != '$']
    
        #save labeled dataset
        self.labeled_dataset = df[['GalaxyID', 'Label']]
        return


    def generate_dataset(self):
        if not self.available_images:
            self.read_available_images()

        if not self.labeled_dataset:
            self.label_dataset()

        #keep a copy of the labeled dataset
        df = self.labeled_dataset.copy()

        #create original filename with jpg extension
        df['orig_name'] = df['GalaxyID'].astype(str) + ".jpg"

        #keep the filenames that exist in the dir and are with labels 0..4
        #Checking labels is not really needed since the labeled dataset should only contain 0...4
        df = df[df['orig_name'].isin(self.available_images)]
    
        #drop >IMG_MAX and create new file name per label
        df_0 = df[df['Label'] == '0'].head(self.max_img[0])
        df_0['counter'] = range(len(df_0))
    
        df_1 = df[df['Label'] == '1'].head(self.max_img[1])
        df_1['counter'] = range(len(df_1))
    
        df_2 = df[df['Label'] == '2'].head(self.max_img[2])
        df_2['counter'] = range(len(df_2))
    
        df_3 = df[df['Label'] == '3'].head(self.max_img[3])
        df_3['counter'] = range(len(df_3))
    
        df_4 = df[df['Label'] == '4'].head(self.max_img[4])
        df_4['counter'] = range(len(df_4))
    
        #concat
        df = pd.concat([df_0, df_1, df_2, df_3, df_4])
    
        #split to train and validation
        train, val, test = np.split(df.sample(frac=1, random_state=self.seed), [int(len(df)*0.8), int(len(df)*0.9)])

        #print(len(train))
        #print(len(val))
        #print(len(test))

        #create new name
        train['new_name'] = "train_class_" + train['Label'] + "_" + train['counter'].astype(str) + ".jpg"
        val['new_name'] = "val_class_" + val['Label'] + "_" + val['counter'].astype(str) + ".jpg"
        test['new_name'] = "test_class_" + test['Label'] + "_" + test['counter'].astype(str) + ".jpg"
    
        #concat
        df = pd.concat([train, val, test]).reset_index(drop=True)

        replica_mapping_records = df[["orig_name", "new_name"]].to_records(index=False)
        replica_mapping_list = list(replica_mapping_records)
        
        return(replica_mapping_list)

        

def main():
    parser = ArgumentParser(description="Enter description here")
    parser.add_argument("-m", "--max_img", metavar=("MAX_IMG_0","MAX_IMG_1","MAX_IMG_2","MAX_IMG_3","MAX_IMG_4"),type=int, nargs=5, default=MAX_IMG_DEFAULT, help="number of instances of each class")
    parser.add_argument("-s", "--seed", type=int, default=10, help="seed for random library")
    parser.add_argument("-d", "--dataset-metadata", type=str, default="training_solutions_rev1.csv", help="Dataset metadata")
    parser.add_argument("-g", "--galaxy-log-file", type=str, default="full_galaxy_data.log", help="Available galaxy images")
    
    args = parser.parse_args()
    
    galaxy_dataset = GalaxyDataset(args.max_img, args.seed, args.dataset_metadata, args.galaxy_log_file)
    l = galaxy_dataset.generate_dataset()
    for k in l:
        print("{0},{1}".format(k[0],k[1]))

    
if __name__ == '__main__':
    main()

