import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
######################################################################################################
## Define your model, transform and all neccessary helper functions here,                           ##
## They will be imported to the exp_recognition.py file                                             ##
######################################################################################################

##Definition of classes as dictionary
classes = {0: 'ANGER', 1: 'DISGUST', 2: 'FEAR', 3: 'HAPPINESS', 4: 'NEUTRAL', 5: 'SADNESS', 6: 'SURPRISE'}

##sample Helper function        
def rgb2gray(image):
    return image.convert('L')

##Sample Transformation function
trnscm=torchvision.transforms.Compose([transforms.Resize((48,48)),transforms.ToTensor()])

# A CNN based Feature extractor
# Definining neural network in python by a class which inherits from nn.Module
