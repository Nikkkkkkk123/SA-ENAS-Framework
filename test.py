# Used corneliases algorithm for image conversion
from PIL import Image
import os
import math
from math import sqrt
import numpy as np
import copy

from torch import nn
import torch

class MalConv(nn.Module):
    def __init__(self,input_length=2000000,window_size=500):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length/window_size))
        

        self.fc_1 = nn.Linear(128,128)
        self.fc_2 = nn.Linear(128,1)

        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()
        

    def forward(self,x):
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x,-1,-2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1,128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        #x = self.sigmoid(x)

        return x

test_folder_path = ""
train_folder_path = ""
testimg_store_path = ""
trainimg_store_path = ""
label_file = ""

# This function maps the label number for the byte file that is stored in the trainLabels.csv file to the actual family name of the malware
# This file can be found on the kaggle dataset page for the Microsoft Malware Classification Challenge
# Paramter: label - the label number as a string
# Return: the family name as a string
def family_folder(label):
    label = label.strip()
    mapping = {
        '1': 'Ramnit',
        '2': 'Lollipop',
        '3': 'Kelihos_ver3',
        '4': 'Vundo',
        '5': 'Simda',
        '6': 'Tracur',
        '7': 'Kelihos_ver1',
        '8': 'Obfuscator.ACY',
        '9': 'Gatak'
    }
    return mapping.get(label, 'Unknown')

# This function opens the label file for the dataset and calls the createImage function for each file
# Parameter: label_file - the path to the label file
# Return: None
def open_label_file(test_folder_path, train_folder_path, testimg_store_path, trainimg_store_path, label_file):
    with open(label_file, 'r') as lf:
        next(lf)  # Skip header line
        for line in lf:
            parts = line.strip().split(',')
            filename = parts[0]
            label = parts[1]
            createImage(filename, label, test_folder_path, train_folder_path, testimg_store_path, trainimg_store_path)

# This function converts the binary string into the pixel values for the image
# Parameter: binaryFile - the binary string representation of the malware
# Return: PixBinFile - list of pixel values
def convertToPixelVal(binaryFile):
    # Take 8 bits to then convert them into a pixel value
    PixBinFile = []
    for start in range(0, len(binaryFile), 8):
        binaryPixel = binaryFile[start:start+8] # get the 8 bits from the string 
        #print(binaryPixel)
        pixelInt =int(binaryPixel, 2) # convert the 8 bits to an integer (2 is base 2)
        #print(pixelInt)
        PixBinFile.append(pixelInt) # append the pixel value to the pixel list
    return PixBinFile

# This function stores the pixel values into the image matrix
# Parameter: image_matrix - the matrix to store the pixel values in
#            matrix_dim - the dimension of the matrix
#            PixBinFile - the list of pixel values
# Return: image_matrix - the matrix with the pixel values stored
def storePixMatri(image_matrix, matrix_dim, PixBinFile):
    cWidth = 0
    cHeight = 0

    for pixel in PixBinFile:
        image_matrix[cHeight][cWidth] = pixel # store the pixel value in the matrix
        cWidth += 1
        if cWidth == matrix_dim: # move to the next row as the current row is now full
            cWidth = 0
            cHeight += 1
    return image_matrix # return the image matrix with all the pixel values now stored

# This function creates the binary string representation of the malware from the byte file
# Parameter: byteFilePath - the path to the byte file
# Return: binaryFile - the binary string representation of the malware
def createBinaryFile(byteFilePath):
    with open (os.path.join(byteFilePath), "rb") as f:
        binaryFile = ""
        counter = 0
        for line in f:
            # Split the line into tokens by spaces
            tokens = line.split()
            
            # The first token is the memory address skip this
            for byte in tokens[1:]:
                counter += 1
                if byte == b'??':  # Handle unknown bytes
                    unknown_value = 'FF'  # Represent unknown bytes as 'FF' (all bits set to 1)
                    binaryByte = format(int(unknown_value, 16), '08b')  # Convert to 8-bit binary string
                    binaryFile += binaryByte
                else:
                    binaryByte = format(int(byte, 16), '08b')  # Convert to 8-bit binary string
                    binaryFile += binaryByte
        appened = copy.deepcopy(binaryFile)
        appened += ("0" * (math.floor(len(binaryFile) * 0.1) - (math.floor(len(binaryFile) * 0.1) % 8)))  # Pad with '1's to make the length a multiple of 8
        return binaryFile

# This function is used to define the global variables required. These variables are hard coded to the current folder locations
def setGlobals():
    global test_folder_path
    global train_folder_path
    global testimg_store_path
    global trainimg_store_path
    global label_file

    test_folder_path = '/media/nikkitanichols/HI/test'
    train_folder_path = "D:\\TrainByte"
    testimg_store_path = '/media/nikkitanichols/HI/TestImg'
    trainimg_store_path = '/media/nikkitanichols/HI/TrainImg'
    label_file = "C:\\Users\\Nikk\\Downloads\\trainLabels.csv"
    return test_folder_path, train_folder_path, testimg_store_path, trainimg_store_path, label_file

def createImage(filename, label, test_folder_path, train_folder_path, testimg_store_path, trainimg_store_path):
    with open(label_file, 'r') as lf:
        next(lf)  # Skip header line
        for line in lf:
            parts = line.strip().split(',')
            filename = parts[0]
            label = parts[1]
            
            # find the filename in the folder_path
            # remove the "" from the start and end of the filename
            filename = filename.strip()
            filename = filename.replace('"', '')
            filename = filename + ".bytes"

            # check both train and test folders
            # if os.path.exists(os.path.join(train_folder_path, filename)):
            #     folder_path = train_folder_path
            #     store_path = trainimg_store_path
            # elif os.path.exists(os.path.join(test_folder_path, filename)):
            #     folder_path = test_folder_path
            #     store_path = testimg_store_path
            # byte_file_path = os.path.join(folder_path, filename)

            # check if the image already exists
            # family = family_folder(label)
            # folder_path_check = os.path.join(store_path, family)
            # if os.path.exists(os.path.join(folder_path_check, filename + '.png')):
            #     #print(f"Image for {filename} already exists in family folder {family}, Folder {folder_path_check}, skipping...")
            #     continue

            # # if this path exists then create the image for it and store it in the correct family folder
            if os.path.exists('D:\\TrainByte\\Obfuscator.ACY\\GFLxvaBSrfoN5uUA3l9s.bytes  '):
                binaryFile = createBinaryFile("D:\\TrainByte\\Obfuscator.ACY\\GFLxvaBSrfoN5uUA3l9s.bytes")
                PixBinFile = convertToPixelVal(binaryFile)

                matrix_dim = sqrt(len(PixBinFile))
                matrix_dim = math.ceil(matrix_dim) # round up to the nearest whole number

                # create an image matrix using the dimensions
                image_matrix = np.zeros((matrix_dim, matrix_dim), dtype=np.uint8) # create a matrix of zeros with the dimensions of the image
                
                image_matrix = storePixMatri(image_matrix, matrix_dim, PixBinFile)
                print(image_matrix[0][0])
                img = Image.fromarray(image_matrix, 'L') # create image from the matrix, 'F' is for (32-bit floating point pixels)
                # name the image the same as the bytes file but with .png extension
                img.show()
                os._exit(0)
            else:
                # the nest 2 lines were for debugging purposes
                #folder_path = family_folder(label)
                #print(folder_path)
                print(f"File {filename}.bytes not found in {folder_path}")
                os._exit(0) 

if __name__ == "__main__":
    test_folder_path, train_folder_path, testimg_store_path, trainimg_store_path, label_file = setGlobals()
    open_label_file(test_folder_path, train_folder_path, testimg_store_path, trainimg_store_path, label_file)
    