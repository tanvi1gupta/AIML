import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
##remove '.' while working on your local system. however make sure that it is present while uploading to the server
from .exp_recognition_model import *
from PIL import Image
import base64
import io
import os
import torch.nn.functional as F
#####################################################################################################################################################
#Caution: Don't change any of the filenames, function names and definitions                                                                         #
#Always use the current_path + file_name for refering any files, without it we cannot access files on the server                                    # 
#####################################################################################################################################################
#Current_path stores absolute path of the directory the file is present. 
current_path = os.path.dirname(os.path.abspath(__file__))
exp_model_location = current_path + '/checkpoint_ckpt_48x48_84epochs_31_7Oct_dict.t7'
#Loading Viola-jones face detector
#Function to detect faces in the image,
#it returns only one image which has maximum area out of all the detected faces in the photo
#If no face is detected, it returns zero(0)
def detected_face(image):
    eye_haar = current_path + '/haarcascade_eye.xml'
    face_haar = current_path + '/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_haar)
    eye_cascade = cv2.CascadeClassifier(eye_haar)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas=[]
    images = []
    required_image=0
    for i, (x,y,w,h) in enumerate(faces):
        face_cropped = gray[y:y+h, x:x+w]
        face_areas.append(w*h)
        images.append(face_cropped)
        required_image = images[np.argmax(face_areas)]
        required_image = Image.fromarray(required_image)
    return required_image
##Image captured from mobile is passed as parameter to this function in the API call, It returns the Expression detected by your network
##The image is passed to the function in base64 encoding, Code to decode the image provided within the function
##Define an object to your network here in the function and load the weight from the trained network, set it in evaluation mode
##Perform neccessary transformations to the input(detected face using the above function), this should return the Expression in string form ex: "Anger" 
##Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_expression(img_str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgdata = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(imgdata))
    img =  np.array(img.getdata()).reshape(img.size[1], img.size[0], 3).astype(np.uint8)
    ##########################################################################################
    ##Example: for loading a model use weight state dictionary                            ##
    ##face_det_net = facExpRec()#Example network                                            ##
    ##face_det_net.load_state_dict(torch.load(current_path + '/exp_recognition_net.stdt')). ##
    ##current_path + '/<network_definition>' is path of the saved model if present in   ##
    ##the same path as this file, we recommend to put in the same directory.                ##
    ##########################################################################################
    face = detected_face(img)
    if face==0:
        return "No Face Found"
    face = trnscm(face)
    #Your code here, return expression using your model
    #feature_net = LeNet()
    feature_net = torch.load(exp_model_location)
    exp_net = LeNet()
    exp_net.load_state_dict(feature_net)
    
    #feature_net = torch.load(exp_model_location, map_location='cpu')
    #exp_net = feature_net['net']
    face = face.reshape(1,1,48,48)
    result = exp_net.forward(face)
    # Coverting the predictions to probabilities, by applying the softmax function
    result = F.softmax(result)
    # Finding the prediction with the largest probability
    _,pred = torch.max(result,1)
    return classes[pred.item()]

# A CNN based Feature extractor
# Definining neural network in python by a class which inherits from nn.Module
class LeNet(nn.Module):
    """LeNet feature extractor model."""

    def __init__(self):
        """Init LeNet feature extractor model."""
        super(LeNet, self).__init__()

        # Defining the CNNfeature Extractor
        self.feature_extractor = nn.Sequential(
            # input [1 x 128 x 128]
            #[1 x 48 x 48]
            # 1st conv layer
            # Conv which convolves input image with 6 filters of 5x5 size, without padding
            nn.Conv2d(1, 6, kernel_size=5),
            # [6 x 124 x 124]
            # [6 x 44 x 44]
            nn.MaxPool2d(kernel_size=2), # Max pooling subsampling operation
            # [6 x 62 x 62]
            # [6 x 22 x 22]
            nn.ReLU(), # Non linear activation function
            # 2nd conv layer
            # input [6 x 62 x 62], [6 x 22 x 22]
            # Conv which convolves input image with 16 filters of 5x5 size, without padding
            nn.Conv2d(6, 16, kernel_size=5),
            # [16 x 58 x 58]
            # [16 x 18 x 18]
            nn.MaxPool2d(kernel_size=2),
            # [16 x 29 x 29]
            #[16 x 9 x 9]
            nn.ReLU()
        )
        
        # Defining the Classifier
        self.classifier = nn.Sequential(
            # Linear layer with 120 nodes, taking a flattened [16 x 4 x 4] as input
            #nn.Linear(16 * 29 * 29, 120),
            nn.Linear(16 * 9 * 9, 120),
            # Linear layer with 84 nodes
            nn.Linear(120, 84),
            # ReLU
            nn.ReLU(),
            # Output layer with as many nodes as number of classes
            nn.Linear(84, 7)
        )
        
    def forward(self, input):
        """Define a Forward pass of the LeNet."""
        out = self.feature_extractor(input) # Pass input through the feature extractor
        #out = out.view(-1, 16 * 29 * 29) # Reshape the 2D to a vector
        out = out.view(-1, 16 * 9 * 9) # Reshape the 2D to a vector
        out = self.classifier(out) # pass features through the classifier to get predictions
        return out
