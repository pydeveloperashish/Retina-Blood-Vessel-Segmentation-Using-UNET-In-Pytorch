import os
import time
import numpy as np
import cv2
import imageio
import torch
import matplotlib.pyplot as plt
from operator import add
from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from unet.unet_model import build_unet
from utils import create_dir, seeding
from logger import logging



def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)         # Flattening
    
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)         # Flattening
    

    score_jaccard = jaccard_score(y_true = y_true, y_pred = y_pred)
    score_f1 = f1_score(y_true = y_true, y_pred = y_pred)
    score_recall = recall_score(y_true = y_true, y_pred = y_pred)
    score_precision = precision_score(y_true = y_true, y_pred = y_pred)
    score_accuracy = accuracy_score(y_true = y_true, y_pred = y_pred)
    
    return [score_jaccard, score_f1, score_recall, score_precision, score_accuracy]



def mask_parse(mask):
    mask = np.expand_dims(mask, axis = -1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis = -1)  ## (512, 512, 3)
    return mask



if __name__ == "__main__":
    
    logging.info(" Testing Started...")
    logging.info("Making Predictions on Test Data...")
    
    """ Seeding """
    seeding(42)
    
    """ Create Directories """
    create_dir("results")
    
    """ Load Dataset """
    AUGMENTED_DATASET_DIR_PATH = os.path.join(os.getcwd(), "augmented_dataset")
    
    test_images = sorted(glob(os.path.join(AUGMENTED_DATASET_DIR_PATH, "test", "images", "*")))
    test_masks = sorted(glob(os.path.join(AUGMENTED_DATASET_DIR_PATH, "test", "masks", "*")))
    
    
    """ Hyperparameters """
    H = 512
    W = 512
    SIZE = (H, W)
    CHECKPOINT_PATH = os.path.join(os.getcwd(), "files", "checkpoint.pth")
    
    
    """ Load the Checkpoint """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location = device))
    
    model.eval()   # Model Eval ON
    
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    
    for i, (x, y) in tqdm(enumerate(zip(test_images, test_masks)), total = len(test_images)):
        
        
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]
        #print(name)
        
        
        """" Reading the image """
        image = cv2.imread(filename = x, flags = cv2.IMREAD_COLOR)  ## (512, 512, 3)
        ## image = cv2.resize(src = image, dsize = SIZE)  -> Already resized.
        x = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis = 0)     ## (1, 3, 512, 512)  Batch Size added.
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)
        
        
        """ Reading the mask """
        mask = cv2.imread(filename = y, flags = cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        ## mask = cv2.resize(src = mask, dsize = SIZE)   -> Already resized.
        y = np.expand_dims(mask, axis = 0)            ## (1, 512, 512)
        y = y / 255.0
        y = np.expand_dims(mask, axis = 0)            ## (1, 1, 512, 512)  Batch Size added.
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)
        
        
        """ Make Prediction """
        with torch.inference_mode():
            start_time = time.time()
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)
            total_time = time.time() - start_time
            time_taken.append(total_time)
            
            score = calculate_metrics(y, y_pred)
            metrics_score = list(map(add, metrics_score, score))
            y_pred = y_pred[0].cpu().numpy()        ## (1, 512, 512)wwwwwwwwww
            y_pred = np.squeeze(y_pred, axis = 0)   ## (512, 512)
            y_pred = y_pred > 0.5
            y_pred = np.array(y_pred, dtype = np.uint8)
            
            
        """ Saving masks """
        ori_mask = mask_parse(mask)
        y_pred = mask_parse(y_pred)
        y_pred = y_pred * 255
        line = np.ones((SIZE[1], 10, 3)) * 128

        image = cv2.putText(img = image, text = "original image", 
                            org = (0, 30), fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale = 1, color = [0, 0, 255], thickness = 2)
        
        ori_mask = cv2.putText(img = ori_mask, text = "original mask", 
                            org = (0, 30), fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale = 1, color = [0, 0, 255], thickness = 2)
        
        y_pred = cv2.putText(img = y_pred, text = "predicted mask", 
                            org = (0, 30), fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale = 1, color = [0, 0, 255], thickness = 2)
        
        
        cat_images = np.concatenate(
            [image, line, ori_mask, line, y_pred], axis = 1
            )
        sample_input_output = np.concatenate(
            [image, line, y_pred], axis = 1
            )
        
        sample_input_output = cv2.resize(sample_input_output, (SIZE))
        
        cv2.imwrite(f"results/{name}.png", cat_images)
        cv2.imwrite(f"sample_result.png", sample_input_output)

    logging.info("Saved the Resulted Mask Images at results folder...")
    
    jaccard = metrics_score[0] / len(test_images)
    f1 = metrics_score[1] / len(test_images)
    recall = metrics_score[2] / len(test_images)
    precision = metrics_score[3] / len(test_images)
    acc = metrics_score[4] / len(test_images)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")
    logging.info(f"Test Metrics are: Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")
    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)
            