#!/usr/bin/env python3

import torch
import argparse
import os
import seaborn as sns
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
import logging
import time
from model_selection import EarlyStopping, VGG16Model
from data_loader import GalaxyDataset


timestr = time.strftime("%Y%m%d-%H%M%S")
###################################################################################################
# Paths:

REL_PATH = ""
DATA_DIR = ""
TRAIN_DATA_PATH  = REL_PATH + DATA_DIR 
TEST_DATA_PATH   = REL_PATH + DATA_DIR
VAL_DATA_PATH    = REL_PATH + DATA_DIR

CHECKPOINT_PATH       = "checkpoint_vgg16.pkl"
EARLY_STOPPING_PATH   = "early_stopping_vgg16_model.pth"
FINAL_CHECKPOINT_PATH = "final_vgg16_model.pth"
VIS_RESULTS_PATH      = REL_PATH + ''

try:
    os.makedirs(VIS_RESULTS_PATH)
except Exception as e:
    pass

# Constant variabless
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = [256, 256]
tensor = (3,256, 256) # this is to predict the in_features of FC Layers

PATIENCE = 4


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
    parser.add_argument('--save', type=str, default = REL_PATH + 'checkpoints/vgg16_galaxy/',help='path to checkpoint save directory ')
    parser.add_argument('--epochs', type=int,default=1, help = "number of training epochs")   
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

# Training loop

def train_loop(model, tloader, vloader, criterion, optimizer):
    """
    returns loss and accuracy of the model for 1 epoch.
    params: model -  vgg16
          tloader - train dataset
          vloader - val dataset
          criterion - loss function
          optimizer - Adam optimizer
    """
    total = 0
    correct = 0
    train_losses = []
    valid_losses = []
    t_epoch_accuracy = 0
    v_epoch_accuracy = 0   
    model.train()

    for sample_batch in tloader:
        image,label   = sample_batch["image"].float(), sample_batch["label"]
     
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(image)
        loss   = criterion(output, label)
        train_losses.append(loss.item())       
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted==label).sum().item()

        loss.backward()
        optimizer.step()

    t_epoch_accuracy = correct/total
    t_epoch_loss = np.average(train_losses)
    
    total = 0
    correct = 0
    y_act = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for sample_batch in vloader:
            
            image,label   = sample_batch["image"].float(), sample_batch["label"]
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)
            loss = criterion(output, label)
            valid_losses.append(loss.item())

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted==label).sum().item()
            y_act.extend(label.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
       
    v_epoch_accuracy = correct/total
    v_epoch_loss = np.average(valid_losses)
            
    return t_epoch_loss, t_epoch_accuracy, v_epoch_loss, v_epoch_accuracy



###################################################################################################  

# Evaluation functions 
def get_all_preds(model, loader):
    """
    returns predictions on test dataset along with their true labels
    params: model = trained model
            loader = test dataloader
    """
    
    all_preds = torch.tensor([]).to(DEVICE)
    all_labels = torch.tensor([]).to(DEVICE)
    
    for batch in loader:
        images, labels = batch["image"].float(), batch["label"]
        preds = model(images.to(DEVICE))
        _, predicted = torch.max(preds.data, 1)
        all_preds = torch.cat(
            (all_preds, predicted),dim = 0 )
        all_labels = torch.cat((all_labels, labels.to(DEVICE)),dim=0)
        
    return all_preds, all_labels




def create_confusion_matrix(model, testloader):
    """
    plots confusion matrix for results on test dataset
    params: model = trained model
            test loader = test dataloader
    """    
    preds, labels = get_all_preds(model, testloader)    
    preds = preds.cpu().tolist()
    labels = labels.cpu().tolist()
    cm = confusion_matrix(labels, preds)    
    skplt.metrics.plot_confusion_matrix(labels, preds, normalize=True)   
    plt.savefig(VIS_RESULTS_PATH + "confusion_matrix_norm.png")
    skplt.metrics.plot_confusion_matrix(labels,preds, normalize=False)
    plt.savefig(VIS_RESULTS_PATH + "confusion_matrix_unnorm.png")

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
    plt.savefig(VIS_RESULTS_PATH + "/confusion_matrix_norm.png")
    plt.close()


def draw_training_curves(train_losses, test_losses, curve_name):
    """
    plots training and testing loss/accuracy curves
    params: train_losses = training loss
            test_losses = validation loss
            curve_name = loss or accuracy
    """
    
    plt.clf()
    max_y = 0
    if curve_name == "accuracy":
        max_y = 1.0
        plt.ylim([0,max_y])
        
    plt.xlim([0,EPOCHS])
    plt.plot(train_losses, label='Training {}'.format(curve_name))
    plt.plot(test_losses, label='Testing {}'.format(curve_name))
    plt.legend(frameon=False)
    plt.savefig(VIS_RESULTS_PATH + "{}_vgg16.png".format(curve_name))

    

def get_data_loader(prefix):
    """
    returns train/test/val dataloaders
    params: flag = train/test/val
    """
    data_transforms  = transforms.Compose([ToTensorRescale()])

    if prefix == "trainval":       
        train_data   = GalaxyDataset( TRAIN_DATA_PATH ,prefix = prefix, use_cache=False,transform = data_transforms)
        train_loader = torch.utils.data.DataLoader(train_data, num_workers = 0, batch_size = BATCH_SIZE, shuffle=True)      
        return train_loader
    
    
    elif prefix == "test":
        test_data   = GalaxyDataset( TEST_DATA_PATH,prefix = prefix,use_cache=False, transform= data_transforms)  
        test_loader = torch.utils.data.DataLoader(test_data, num_workers = 0, batch_size = BATCH_SIZE, shuffle=True)       
        return test_loader
    
###################################################################################################  


def train_model(best_params):
    
    print("Training model")
    
    train_loader = get_data_loader("trainval")
    val_loader   = get_data_loader("test")

    layer   = best_params["layer"]
    lr_body = best_params["lr_body"]
    lr_head = best_params["lr_head"]
    
    model     = VGG16Model(layer).to(DEVICE)    
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam([{'params': model.body.parameters(), 'lr':lr_body},
                                 {'params':model.head.parameters(), 'lr':lr_head}])
    
    train_loss = []
    val_loss   = []
    train_acc  = []
    val_acc    = []
    total_loss  = 0
    start_epoch = 0
    restart = False
    
    early_stop = EarlyStopping(patience=PATIENCE, path= EARLY_STOPPING_PATH)


    try:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer)
        print("Checkpoint loaded. Restarting training.")
        restart = True
        print(start_epoch)
    except Exception as e:
        print("Checkpoint not found. Training from scratch.")
    
    if start_epoch < EPOCHS:

        for epoch in range(start_epoch, EPOCHS):
            print("Running Epoch {}".format(epoch+1))
            if (epoch > 0) and (not restart):
                train_loader.dataset.set_use_cache(use_cache=True)
                val_loader.dataset.set_use_cache(use_cache=True)

            epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc = train_loop(model, train_loader, val_loader, criterion, optimizer)
            train_loss.append(epoch_train_loss)
            train_acc.append(epoch_train_acc)
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_acc)
            total_loss += epoch_val_loss
            print("Training loss: {0:.4f}  Train Accuracy: {1:0.2f}".format(epoch_train_loss, epoch_train_acc))
            print("Validation loss: {0:.4f}  Validation Accuracy: {1:0.2f}".format(epoch_val_loss, epoch_val_acc))
            print("--------------------------------------------------------")
           
            early_stop(epoch_val_loss, model, optimizer, epoch, layer)
            restart = False
        
            if early_stop.early_stop:
                os.rename(EARLY_STOPPING_PATH, FINAL_CHECKPOINT_PATH)
                break

            if epoch % 2 == 0:
                save_checkpoint(model, optimizer, epoch,layer, CHECKPOINT_PATH)

        
        draw_training_curves(train_loss, val_loss, "loss")
        
        if not early_stop.early_stop:
            save_checkpoint(model, optimizer, epoch,layer,FINAL_CHECKPOINT_PATH)
        
        total_loss/=EPOCHS
    else:
        print("Training has been already completed.")
   
    return


def load_checkpoint(model, optimizer):
    
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer,epoch



def save_checkpoint(model, optimizer, epoch, layer,checkpoint_path):
    torch.save({
            'epoch': epoch,
            'layer': layer,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)



    
def main():
    
    start = time.time()
    
    global ARGS
    global BATCH_SIZE
    global EPOCHS

    ARGS = get_arguments()   
    seed = ARGS.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if (ARGS.cuda):
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False 

    BATCH_SIZE = ARGS.batch_size
    EPOCHS     = ARGS.epochs
    
    f = open("best_vgg16_hpo_params.txt").read()
    
    best_params = eval(f)
    best_params = best_params["params"]

    train_model(best_params)

    exec_time = time.time() - start

    print('Execution time in seconds: ' + str(exec_time))
    return

if __name__ == "__main__":
    
    main()
