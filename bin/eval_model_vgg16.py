#!/usr/bin/env python3
import torch
import argparse
import os
import sys
import seaborn as sns
import numpy as np
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import logging
import time
from model_selection import EarlyStopping, VGG16Model
from data_loader import GalaxyDataset


timestr = time.strftime("%Y%m%d-%H%M%S")

###################################################################################################
# PATHS
REL_PATH = ""
DATA_DIR = ""
TEST_DATA_PATH        = REL_PATH + DATA_DIR 
VIS_RESULTS_PATH      = REL_PATH + ''
FINAL_CHECKPOINT_PATH = "final_vgg16_model.pth"
results_record = open(VIS_RESULTS_PATH+"exp_results.csv", 'w+')

try:
    os.makedirs(VIS_RESULTS_PATH)
except Exception as e:
    pass

# Constant variabless
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = [256, 256]
tensor   = (3,256, 256) # this is to predict the in_features of FC Layers

# TO ADD if memory issues encounter
#gc.collect()
#torch.cuda.empty_cache()

### ------------------------- LOGGER--------------------------------
logger = logging.getLogger('optuna_db_log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_arguments():
    
    parser = argparse.ArgumentParser(description="Galaxy Classification")   
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--cuda', type=int, default=0, help='use gpu support')
    parser.add_argument('--seed', type=int, default=123, help='select seed number for reproducibility')
    parser.add_argument('--root_path', type=str, default='./data',help='path to dataset ')   
    args = parser.parse_args()
    
    return args


### -------------------------FOR DATALOADER --------------------------------
class ToTensorRescale(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = image/255
        image = np.resize(image,(256,256,3))
        image = image.transpose((2, 0, 1))
        return {"image":torch.from_numpy(image),
                "label" :label}


###################################################################################################        




def run_inference(model,  vloader):
    total   = 0
    correct = 0
    y_act   = []
    y_pred  = []

    model.eval()
    with torch.no_grad():
        for sample_batch in vloader:
            
            image,label   = sample_batch["image"].float(), sample_batch["label"]
            image  = image.to(DEVICE)
            label  = label.to(DEVICE)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted==label).sum().item()
            y_act.extend(label.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
       
    accuracy = correct/total
    prec, recall, fscore, _ = precision_recall_fscore_support(y_act, y_pred, average='macro')
    results_record.write("Acc: {}, Prec: {}, recall: {}, fscore: {} \n".format(accuracy, prec, recall, fscore))
    cm = confusion_matrix(y_act, y_pred)
    results_record.write(str(cm))
    plot_cm(y_act, y_pred)

    return



###################################################################################################  
def plot_cm(lab, pred):
    target_names = ["0","1","2","3", "4"]
    cm = confusion_matrix(lab, pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, cmap = "YlGnBu")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("VGG-16")
    plt.savefig(VIS_RESULTS_PATH + "final_confusion_matrix_norm.png")
    plt.close()




def get_data_loader(prefix):
    """
    returns train/test/val dataloaders
    params: flag = train/test/val
    """
    data_transforms  = transforms.Compose([ToTensorRescale()])
    
    if prefix == "test":
        test_data   = GalaxyDataset( TEST_DATA_PATH,prefix = prefix,use_cache=False, transform= data_transforms)  
        test_loader = torch.utils.data.DataLoader(test_data, num_workers = 0, batch_size = BATCH_SIZE, shuffle=True)       
        return test_loader
    
###################################################################################################  


def test_model(best_params):
    
    val_loader = get_data_loader("test")    
    model      = VGG16Model(best_params["layer"]).to(DEVICE)    

    try:
        model  = load_checkpoint(model)
        run_inference(model, val_loader)
        print("Read in the weights with the model.")
    except Exception as e:
        print(e)
    return


def load_checkpoint(model):
    #try:
    checkpoint = torch.load(FINAL_CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    #except Exception as e:
    #    print(e)
    return model


def main():
    
    start = time.time()
    
    global ARGS
    global BATCH_SIZE

    ARGS = get_arguments()   
    BATCH_SIZE = ARGS.batch_size
    f = open("best_vgg16_hpo_params.txt").read()
    
    best_params = eval(f)
    best_params = best_params["params"]

    test_model(best_params)

    exec_time = time.time() - start

    print('Execution time in seconds: ' + str(exec_time))
    return

if __name__ == "__main__":
    
    main()
