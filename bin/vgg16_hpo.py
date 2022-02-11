#!/usr/bin/env python3
import torch
import argparse
import optuna
import joblib
import numpy as np
import time
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
import logging
from model_selection import EarlyStopping, VGG16Model
from data_loader import GalaxyDataset


timestr = time.strftime("%Y%m%d-%H%M%S")
###################################################################################################
# Paths:

REL_PATH = "./"
DATA_DIR = "./"
TRAIN_DATA_PATH  = REL_PATH + DATA_DIR 
TEST_DATA_PATH   = REL_PATH + DATA_DIR
VAL_DATA_PATH    = REL_PATH + DATA_DIR
CHECKPOINT_DIR   = REL_PATH 
VIS_RESULTS_PATH = REL_PATH 



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
    parser.add_argument('--trials', type=int, default=2, help = "number of HPO trials")
    
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
    
    preds  = preds.cpu().tolist()
    labels = labels.cpu().tolist()
    cm     = confusion_matrix(labels, preds)
    
    skplt.metrics.plot_confusion_matrix(labels, preds, normalize=True)
    
    plt.savefig(VIS_RESULTS_PATH + "/confusion_matrix_norm.png")
    skplt.metrics.plot_confusion_matrix(labels,preds, normalize=False)
    plt.savefig(VIS_RESULTS_PATH + "/confusion_matrix_unnorm.png")
    plt.close()

    
def draw_training_curves(train_losses, test_losses, curve_name, trial_num):
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
    plt.savefig(VIS_RESULTS_PATH + "/{}_vgg16_trial_{}.png".format(curve_name, trial_num))
    plt.close()

    

def get_data_loader(prefix):
    """
    returns train/test/val dataloaders
    params: flag = train/test/val
    """

    data_transforms = transforms.Compose([ToTensorRescale()])

    if prefix == "train":       

        train_data  = GalaxyDataset( TRAIN_DATA_PATH ,prefix = prefix,use_cache=False, transform = data_transforms)
        train_loader = torch.utils.data.DataLoader(train_data, num_workers = 0, batch_size = BATCH_SIZE, shuffle=True)       
        return train_loader
    
    elif prefix == "val":
        
        val_data   = GalaxyDataset( VAL_DATA_PATH, prefix = prefix, use_cache=False,transform= ToTensorRescale())
        val_loader = torch.utils.data.DataLoader(val_data,num_workers = 0, batch_size = BATCH_SIZE, shuffle=True)
        return val_loader
    
    elif prefix == "test":

        test_data   = GalaxyDataset( TEST_DATA_PATH,prefix = prefix,use_cache=False, transform= ToTensorRescale())  
        test_loader = torch.utils.data.DataLoader(test_data, num_workers = 0, batch_size = BATCH_SIZE, shuffle=True)       
        return test_loader
    
###################################################################################################  

# Optuna Study functions

def objective(trial,direction = "minimize"):
    
    print("Performing trial {}".format(trial.number))
    
    train_loader = get_data_loader("train")
    val_loader   = get_data_loader("val")


    
    layer     = trial.suggest_categorical("layer",["21"])    
    model     = VGG16Model(layer).to(DEVICE)    
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    lr_body   = trial.suggest_categorical("lr_body", [1e-7, 1e-8, 1e-9])
    lr_head   = trial.suggest_categorical("lr_head", [1e-4, 1e-5, 1e-6])
    optimizer = torch.optim.Adam([{'params': model.body.parameters(), 'lr':lr_body},
                                 {'params':model.head.parameters(), 'lr':lr_head}])
    
    train_loss = []
    val_loss   = []
    train_acc  = []
    val_acc    = []
    total_loss = 0
    
    early_stop = EarlyStopping(patience=PATIENCE, path= CHECKPOINT_DIR+'/early_stopping_vgg16_model_trial{}.pth'.format(trial.number))
    
    for epoch in range(EPOCHS):
        print("Running Epoch {}".format(epoch+1))
        if epoch > 0:
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
    
        if early_stop.early_stop:
            break
    draw_training_curves(train_loss, val_loss, "loss", trial.number )
    
    
    total_loss/=EPOCHS
    
    return total_loss
    
    
def hpo_monitor(study, trial):
    """
    Save optuna hpo study
    """
    joblib.dump(study, CHECKPOINT_DIR+"hpo_galaxy_vgg16.pkl")
    



def get_best_params(best):
    """
    Saves best parameters of Optuna Study.
    """
    
    parameters = {}
    parameters["trial_id"] = best.number
    parameters["value"] = best.value
    parameters["params"] = best.params
    f = open("best_vgg16_hpo_params.txt","w")
    f.write(str(parameters))
    f.close()


#####
# Checkpointing we can restart existing study
####
def create_optuna_study():

    global STUDY
    load_checkpoint_flag = True
    try:
        STUDY = joblib.load("hpo_galaxy_vgg16.pkl")
        todo_trials = TRIALS - len(STUDY.trials_dataframe())
        if todo_trials > 0 :
            print("There are {} trials to do out of {}".format(todo_trials, TRIALS))
            STUDY.optimize(objective, n_trials=todo_trials,  callbacks=[hpo_monitor])
            best_trial = STUDY.best_trial
            get_best_params(best_trial)
        else:
            print("This study is finished. Nothing to do.")
    except Exception as e:
        STUDY = optuna.create_study(direction = 'minimize', study_name='Galaxy Classification')
        STUDY.optimize(objective, n_trials= TRIALS,  callbacks=[hpo_monitor])
        best_trial = STUDY.best_trial
        get_best_params(best_trial)

    
    
def main():

    start = time.time()   
    global TRIALS
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

    TRIALS     = ARGS.trials
    BATCH_SIZE = ARGS.batch_size
    EPOCHS     = ARGS.epochs

    create_optuna_study()
    exec_time = time.time() - start


    print('Execution time in seconds: ' + str(exec_time))
    return

if __name__ == "__main__":
    
    main()
