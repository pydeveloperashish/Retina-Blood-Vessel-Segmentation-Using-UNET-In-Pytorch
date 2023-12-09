from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
from PIL import Image
import base64
import io
from werkzeug.utils import secure_filename
import cv2
import torch

from components.unet.unet_model import build_unet
from components.utils import get_model_from_gdrive

app = Flask(__name__)


app.config["IMAGE_UPLOADS"] = os.path.join(os.getcwd(), "static", "uploads")



def mask_parse(mask):
    mask = np.expand_dims(mask, axis = -1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis = -1)  ## (512, 512, 3)
    return mask



@app.route('/', methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        image = request.files['file']

        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        
        if not os.path.exists(app.config["IMAGE_UPLOADS"]):      
            os.makedirs(app.config["IMAGE_UPLOADS"])
            print(f"Directory {app.config['IMAGE_UPLOADS']} created successfully.")
            
        image.save(os.path.join(
            basedir, app.config["IMAGE_UPLOADS"], filename))
        
        # Reading the image into PIL format.
        pil_img = Image.open(app.config["IMAGE_UPLOADS"] + "/" + filename)
        
        # Converting it into opencv format to do some operations
        opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        

        """ Hyperparameters """
        H = 512
        W = 512
        SIZE = (H, W)
        CHECKPOINT_PATH = os.path.join(os.getcwd(), "files", "checkpoint.pth")

        """ This below lines of checking CHECKPOINT_PATH and its Size in MB is added later """
        if not os.path.exists(CHECKPOINT_PATH) or os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024) < 100:
            print("Downloading the checkpoint model")
            get_model_from_gdrive()
            print('Checkpoint model downloaded successfully...')
            CHECKPOINT_PATH = os.path.join(os.getcwd(), "files", "checkpoint.pth")
        
        """ Load the Checkpoint """
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_unet()
        model = model.to(device)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location = device))
        print('Checkpoint model found')

        """" Reading the image """
        x = cv2.resize(src = opencvImage, dsize = SIZE)
        # print(x.shape)
        x = np.transpose(x, (2, 0, 1))  # (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis = 0)  # (1, 3, 512, 512)  Batch Size added.
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)


        """ Make Prediction """
        with torch.inference_mode():
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred[0].cpu().numpy()  # (1, 512, 512)wwwwwwwwww
            y_pred = np.squeeze(y_pred, axis = 0)  # (512, 512)
            y_pred = y_pred > 0.5
            y_pred = np.array(y_pred, dtype = np.uint8)


        """ Saving masks """
        y_pred = mask_parse(y_pred)
        y_pred = y_pred * 255
        line = np.ones((SIZE[1], 10, 3)) * 128

        opencvImage = cv2.putText(img = opencvImage, text = "original image",
                            org = (0, 30), fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale = 1, color = [0, 0, 255], thickness = 2)

        y_pred = cv2.putText(img = y_pred, text = "predicted mask",
                            org = (0, 30), fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale = 1, color = [0, 0, 255], thickness = 2)

        
        cat_images = np.concatenate(
            [opencvImage, line, y_pred], axis = 1
        )

        filename = filename.split(".")[0]
        resulted_filename = f"{filename}_result.png"
        
        
        cv2.imwrite(os.path.join(os.getcwd(), "static", "uploads",
                    resulted_filename), cat_images)
        
        
        
        # Reading the Resulted Image into PIL format.
        pil_img = Image.open(os.path.join(app.config["IMAGE_UPLOADS"], resulted_filename))
        
        data = io.BytesIO()
        pil_img.save(data, "png")
        
        encode_img_data = base64.b64encode(data.getvalue())

        return render_template("main.html", filename = encode_img_data.decode("UTF-8"))

    return render_template('main.html')



if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080)
