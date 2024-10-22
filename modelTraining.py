import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as tf
from torch.utils.data import DataLoader, Dataset
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from PIL import Image
from model import MaskModel
import os

DEVICE = "cpu"
LEARNING_RATE = .001
BATCH_SIZE = 16
EPOCHS = 10
TRAINING_IMG_DIR = "data/training_raw/"
TRAINING_MASK_DIR = "data/training_mask/"
TESTING_IMG_DIR = "data/testing_raw/"
TESTING_MASK_DIR = "data/testing_mask/"
NUM_WORKERS = 2 #Seeing if this works with windows if within mani function


class MaskDataset(Dataset):
    def __init__(self, imgDir, maskDir, transforms):#Creating custom dataset
        super().__init__()

        self.imgDir = imgDir
        self.maskDir = maskDir
        self.transforms = transforms
        self.images = os.listdir(imgDir)
        self.masks = os.listdir(maskDir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgPath = os.path.join(self.imgDir, self.images[index])
        maskPath = os.path.join(self.imgDir, self.masks[index])
        img = np.array(Image.open(imgPath).convert('RGB'), dtype= np.float32)#Might have to convert to RGB
        maskPath = maskPath.replace("Masked", "Unmasked")
        mask = np.array(Image.open(maskPath).convert('RGB'), dtype=np.float32)#Will give gray scale but we don't want gray scale




        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        
        return img, mask





def main():
    def save_imgs():
        model.eval()
        for i, (X,y) in enumerate(testingLoader):
            X = X.to(DEVICE)
            with torch.no_grad():
                preds = model(X)
                preds = (preds > 0.5).float()#Also inaccurate as it will produce as it will be binary
            torchvision.utils.save_image(preds, f"results/prediction{i}.png")
            torchvision.utils.save_image(y.unsqueeze(1), f"results/ground_truth{i}.png")





    transforms = alb.Compose([alb.Resize(height=256, width=256), ToTensorV2()])#The values of height and width are not accurate
    
    model = MaskModel(inputChannels=3, outputChannels=3, sizes=[32, 64, 128, 256]).to(device=DEVICE)#Remember that out channels = 3 because of the 3 classes
    lossFunction = nn.CrossEntropyLoss(ignore_index=255)#might have to add "sigmoid" on forward pass in model.py
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    

    trainigData = MaskDataset(imgDir=TRAINING_IMG_DIR, maskDir=TRAINING_MASK_DIR, transforms=transforms)
    testingData = MaskDataset(imgDir=TESTING_IMG_DIR, maskDir=TESTING_MASK_DIR, transforms=None)#Get the data and transform them

    trainingLoader = DataLoader(trainigData, num_workers=NUM_WORKERS, shuffle=True)
    testingLoader = DataLoader(testingData, num_workers=NUM_WORKERS, shuffle=False)

    print("\n\n\n started training \n\n\n")
    for epoch in range(EPOCHS):
        model.train()
        for batch, (X, y) in enumerate(trainingLoader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            preds = model(X)
            print(f"pred shape = {preds.shape}, y shape = {y.shape}\n\n\n")
            loss = lossFunction(preds, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        numCorrect= 0
        numPixels = 0
        model.eval()
        with torch.no_grad():#Get rid of the gradients
            for X, y in testingLoader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X)
                preds = torch.sigmoid(preds)#convert to probabilities
                preds = (preds > 0.5).float() #need to change for multiclass and need to implements intersection over union
                numCorrect += (preds == y).sum()
                numPixels += torch.numel(preds)
            print(f"finished {epoch} with acc{numCorrect/numPixels*100:.2f}")
        
    save_imgs()#will save to results directory


if __name__ == "__main__":
    main()






