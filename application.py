from flask import Flask,request,render_template;
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle


app= Flask(__name__)

APP_ROUTE=os.path.dirname(os.path.abspath(__file__))

def convert_jpg_to_tif(jpg_path, tif_path):
    
    # Open the .jpg image
    with Image.open(jpg_path) as img:
        # Convert the image to RGB mode if it's not already in that mode
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Save the image as .tif
    img.save(tif_path, "TIFF")

def convert_tif_to_jpg(tiff_path, jpg_path):
    try:
        # Open the TIFF file
        with Image.open(tiff_path) as img:
            # Check if the image has an alpha channel (transparency)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                # Convert image to RGB mode (remove alpha channel)
                img = img.convert("RGB")
            else:
                # Convert to RGB mode if not already in a compatible format
                img = img.convert("RGB")
            
            # Save the image as JPG
            img.save(jpg_path, "JPEG")
        print(f"Successfully converted {tiff_path} to {jpg_path}")
    except Exception as e:
        print(f"Error converting {tiff_path} to JPG: {e}")


@app.route("/")
def mainpage():
    return render_template("index3.html")


def tumor_detect(image_path):
    # Define the custom dice coefficient function
    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # Load the model architecture from JSON
    with open('segmentation_best_model_part_3.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={'dice_coef': dice_coef})

    # Load the model weights
    model.load_weights('segmentation_best_model_part_3.weights.h5')

    def preprocess_image(image_path):
    # Load the image with the target size
        img = load_img(image_path, target_size=(256, 256))
    # Convert the image to array
        img_array = img_to_array(img)
    # Normalize the image
        img_array = img_array / 255.0
    # Expand dimensions to match the model input shape
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def postprocess_prediction(prediction):
        # Remove the batch dimension
        prediction = np.squeeze(prediction, axis=0)
        # Apply a threshold to convert to binary image
        prediction = (prediction > 0.5).astype(np.uint8)
        return prediction

    def predict_segmentation(image_path):
        # Preprocess the input image
        preprocessed_image = preprocess_image(image_path)
        # Make prediction
        prediction = model.predict(preprocessed_image)
        # Postprocess the prediction
        processed_prediction = postprocess_prediction(prediction)
        return processed_prediction

    def display_image(image, title='Image'):
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def save_image(image, filepath, title='Image'):
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.savefig(filepath)
        plt.close()

# Function to handle file upload and prediction
    def predict_and_display(image_path):
        # Make a prediction
        prediction = predict_segmentation(image_path)
        # Display the input image
        input_image = load_img(image_path)
        display_image(input_image, title='Input Image')
        # Display the predicted segmentation
        display_image(prediction, title='Predicted Segmentation')

# Example usage
# tif_path = 'path/to/your/input_image.tif'
# jpg_path = 'path/to/your/output_image.jpg'
# convert_tif_to_jpg(tif_path, jpg_path)
    
    def predict_and_save(image_path, save_path):
        # Make a prediction
        prediction = predict_segmentation(image_path)
        # Save the predicted segmentation
        save_image(prediction, save_path, title='Predicted Segmentation')
    # image_path=tumor_detect()
    save_path='static/result.tif'
    # predict_and_display(image_path)
    predict_and_save(image_path,save_path)




@app.route("/output", methods=['GET','POST'])
def output_page():
    image=request.files.get('file')
    filename=image.filename
    print("The gotten image file in the post request: ", filename);
    destination=os.path.join(APP_ROUTE,"static")
    if not os.path.isdir(destination):
        os.mkdir(destination)
    
    image.save("/".join([destination,"temp.tif"]))
    print("The file path ot save the incoming image to: ",destination)


    # convert_jpg_to_tif('static/temp.jpg','static/temp.tif')
    tumor_detect('static/temp.tif')
    convert_tif_to_jpg('static/temp.tif', 'static/temp.jpg')
    convert_tif_to_jpg('static/result.tif', 'static/result.jpg')


    return render_template("index.html", tasks=filename)

@app.route('/risk_page')
def risk_page():
    return render_template('index2.html')


@app.route('/risk_factor', methods=['GET'])
def risk_factor():
    def predict(RNASeqCluster,MethylationCluster, miRNACluster, CNCluster,RPPACluster, OncosignCluster, COCCluster, histological_type,neoplasm_histologic_grade,tumor_tissue_site,laterality,tumor_location,gender,age_at_initial_pathologic,race,ethnicity):
        with open('support_vector_model_1.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        data = [[RNASeqCluster,MethylationCluster, miRNACluster,CNCluster,RPPACluster,OncosignCluster, COCCluster, histological_type,neoplasm_histologic_grade,tumor_tissue_site,laterality,tumor_location,gender,age_at_initial_pathologic,race,ethnicity]]

        return loaded_model.predict(data)
    

    RNASeqCluster=int(request.args.get('RNASeqCluster',default=None,type=str))
    MethylationCluster=int(request.args.get('MethylationCluster',default=None,type=str))
    miRNACluster=int(request.args.get('miRNACluster',default=None,type=str))
    CNCluster=int(request.args.get('CNCluster',default=None,type=str))
    RPPACluster=int(request.args.get('RPPACluster',default=None,type=str))
    OncosignCluster=int(request.args.get('OncosignCluster',default=None,type=str))
    COCCluster=int(request.args.get('COCCluster',default=None,type=str))
    neoplasm_histologic_grade=int(request.args.get('neoplasm_histologic_grade',default=None,type=str))
    tumor_tissue_site=int(request.args.get('tumor_tissue_site',default=None,type=str))
    laterality=int(request.args.get('laterality',default=None,type=str))
    tumor_location=int(request.args.get('tumor_location',default=None,type=str))
    gender=int(request.args.get('gender',default=None,type=str))
    age_at_initial_pathologic=int(request.args.get('age_at_initial_pathologic',default=None,type=str))
    race=int(request.args.get('race',default=None,type=str))
    ethnicity=int(request.args.get('ethnicity',default=None,type=str))
    histological_type=int(request.args.get('histological_type',default=None,type=str))
    
    
    data = [[RNASeqCluster,MethylationCluster, miRNACluster,CNCluster,RPPACluster,OncosignCluster, COCCluster, histological_type,neoplasm_histologic_grade,tumor_tissue_site,laterality,tumor_location,gender,age_at_initial_pathologic,race,ethnicity]]

    print("The data provided by the request: ", data)

    risk_factor=predict(RNASeqCluster,MethylationCluster, miRNACluster, CNCluster,RPPACluster, OncosignCluster, COCCluster, histological_type,neoplasm_histologic_grade,tumor_tissue_site,laterality,tumor_location,gender,age_at_initial_pathologic,race,ethnicity)

    if risk_factor==1:
        risk_factor='The patietnt might be heavily affected and there is high chance of cancer'
    else:
        risk_factor='Low risk of death. But should be checked immediately.'
    return render_template("index2.html", tasks=risk_factor)

if(__name__=="__main__"):
    app.run(debug=True)
    # tumor_detect()