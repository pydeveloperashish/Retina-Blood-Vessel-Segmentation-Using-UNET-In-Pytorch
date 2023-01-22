import os
import numpy as np
import cv2
import imageio
from tqdm import tqdm
from glob import glob
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from utils import create_dir



def load_data(path):
    train_images = glob(os.path.join(path, 'train', 'images', "*.tif"))
    train_mask = glob(os.path.join(path, 'train', 'masks', "*.gif"))
    
    test_images = glob(os.path.join(path, 'test', 'images', "*.tif"))
    test_mask = glob(os.path.join(path, 'test', 'masks', "*.gif"))

    return (train_images, train_mask), (test_images, test_mask)



def augment_data(images, masks, save_path, augment = True):
    
    SIZE = (512, 512)
    
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total = len(images)):
        """ Extracting the name of the image """
        name = x.split("/")[-1].split('.')[0]
    
         
        """ Reading image and mask """
        x = cv2.imread(filename = x, flags = cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0] 
        
        
        if augment == True:
            
            aug = HorizontalFlip(p = 1.0)
            augmented = aug(image = x, mask = y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
            
            aug = VerticalFlip(p = 1.0)
            augmented = aug(image = x, mask = y)
            x2 = augmented["image"]
            y2 = augmented["mask"]
            
            aug = Rotate(limit = 45, p = 1.0)
            augmented = aug(image = x, mask = y)
            x3 = augmented["image"]
            y3 = augmented["mask"]
            
            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
            
            
        else:
            X = [x]
            Y = [y]
            
            
            
        index = 0
            
        for i, m in zip(X, Y):
            i = cv2.resize(src = i, dsize = SIZE)
            m = cv2.resize(src = m, dsize = SIZE)
            
            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"
            
            cv2.imwrite(filename = os.path.join(save_path, "images", tmp_image_name),
                        img = i)
            cv2.imwrite(filename = os.path.join(save_path, "masks", tmp_mask_name),
                        img = m)       
            
            index += 1
            
        
            

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    
    
    """ Load the data """
    DATAPATH = os.path.join(os.getcwd(), 'dataset')
    (train_images, train_mask), (test_images, test_mask) = load_data(DATAPATH)
    
    
    """ Create Directories to save the augmented data """
    create_dir(os.path.join(os.getcwd(), "augmented_dataset", "train", "images"))
    create_dir(os.path.join(os.getcwd(), "augmented_dataset", "train", "masks"))
    create_dir(os.path.join(os.getcwd(), "augmented_dataset", "test", "images"))
    create_dir(os.path.join(os.getcwd(), "augmented_dataset", "test", "masks"))
    
    
    
    """ Data Augmentation """
    
    # For Training Data, we apply Data Augmentation.
    augment_data(images = train_images, 
                 masks = train_mask, 
                 save_path = os.path.join(os.getcwd(), "augmented_dataset", "train"), 
                 augment = True)
    
    # For testing Data, we dont apply Data Augmentation, just resizing it.
    augment_data(images = test_images, 
                 masks = test_mask, 
                 save_path = os.path.join(os.getcwd(), "augmented_dataset", "test"), 
                 augment = False)
    
    