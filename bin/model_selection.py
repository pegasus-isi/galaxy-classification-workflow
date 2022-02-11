#!/usr/bin/env python3
import torch
import torchvision
import numpy as np

tensor = (3,256, 256)

# VGG16trained model Architecture

class VGG16Model(torch.nn.Module):
    """
    VGG16 pretrained model with additional projection head for transfer learning
        
    """
    def __init__(self, layer):
        super(VGG16Model, self).__init__()
        
        self.layer = layer
        self.body = torchvision.models.vgg16(pretrained=True).features
        
        for name,child in self.body.named_children():
            if name == self.layer:
               
                break
            for params in child.parameters():
                params.requires_grad = False
            
      
        self.in_feat = self.get_dim(tensor)

        self.head = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(in_features=self.in_feat, out_features=512, bias=True), #not such a steep jump
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(in_features= 512, out_features=64, bias=True), #not such a steep jump
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(64,5)
        )

    def get_dim(self, input_size):
        bs = 1
        ip = torch.rand(bs, *input_size)
        output = self.body(ip)
        op_view = output.view(bs,-1)
        return op_view.shape[-1]
        
    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


###################################################################################################

# Early stopping implementation

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='early_stopping.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'early_stopping_vgg16model.pth'   
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model, optimizer, epoch, layer):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, optimizer, epoch, layer)
            self.val_loss_min = val_loss
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
            self.save_checkpoint(model, optimizer, epoch, layer)
            self.val_loss_min = val_loss
            self.counter = 0   
    
    def save_checkpoint(self, model, optimizer, epoch, layer):
        """
        saves the current best version of the model if there is decrease in validation loss
        """
        torch.save({
            'epoch': epoch,
            'layer': layer,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, self.path)
        
