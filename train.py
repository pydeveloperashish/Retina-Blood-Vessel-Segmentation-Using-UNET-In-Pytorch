import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob
from utils import seeding, create_dir, epoch_time
from unet.unet_model import build_unet
from unet.loss import DiceLoss, DiceBCELoss
from data import RetinaDataset



def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    
    model.train()    # Training Mode ON
    for x, y in loader:
        x = x.to(device, dtype = torch.float32)
        y = y.to(device, dtype = torch.float32)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss = epoch_loss / len(loader)
    
    return epoch_loss



def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    
    model.eval()    # Eval Mode ON
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, dtype = torch.float32)
            y = y.to(device, dtype = torch.float32)
            
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss / len(loader)
        
    return epoch_loss    







if __name__ == "__main__":
    """ Seeding """
    seeding(42)
    
    """ Create Directories """
    create_dir("files")
    
    
    """ Load New Augmented Dataset """
    AUGMENTED_DATASET_DIR_PATH = os.path.join(os.getcwd(), "augmented_dataset")
    
    train_images = sorted(glob(os.path.join(AUGMENTED_DATASET_DIR_PATH, "train", "images", "*")))
    train_masks = sorted(glob(os.path.join(AUGMENTED_DATASET_DIR_PATH, "train", "masks", "*")))
    
    valid_images = sorted(glob(os.path.join(AUGMENTED_DATASET_DIR_PATH, "test", "images", "*")))
    valid_masks = sorted(glob(os.path.join(AUGMENTED_DATASET_DIR_PATH, "test", "masks", "*")))
    

    """ Hyperparameters """
    H= 512
    W = 512
    SIZE = (H, W)
    BATCH_SIZE = 2
    NUM_EPOCHS = 5
    LR = 0.001
    CHECKPOINT_PATH = os.path.join(os.getcwd(), "files", "checkpoint.pth")
    
    
    """ Dataset and DataLoader """
    train_dataset = RetinaDataset(images_path = train_images, masks_path = train_masks)
    valid_dataset = RetinaDataset(images_path = valid_images, masks_path = valid_masks)
    
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = os.cpu_count()
        )
    
    valid_loader = DataLoader(
        dataset = valid_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = os.cpu_count()
        )
    
    
    """ Setup Device and Model """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_unet()
    model = model.to(device)
    
    
    """ Setup Optimizer, Scheduler and Loss """
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer, mode = "min", 
                patience = 5, verbose = True)
    loss_fn = DiceBCELoss()
    
    
    """ Training the Model """
    for epoch in range(0, NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model = model, loader = train_loader, 
                           optimizer = optimizer, loss_fn = loss_fn, 
                           device = device)
        
        valid_loss = evaluate(model = model, loader = valid_loader, 
                              loss_fn = loss_fn, device = device)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time = start_time, 
                                            end_time = end_time)
        
        
        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)