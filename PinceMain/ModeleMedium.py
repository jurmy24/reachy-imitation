"""Modèle de détection main ouverte ou fermé.
Modèle le plus lourd"""

import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset, Subset
from torchinfo import summary

from torchvision import datasets, transforms
import uuid
import imageio
from PIL import Image

import csv

try:
    from HandGestureUtils import *
except:
    pass

try:
    from PinceMain.HandGestureUtils import *
except:
    pass

def former_dataset(liste_data, shuffle, cat):
    indices = torch.randperm(len(liste_data)).tolist()
    data = Subset(liste_data, indices)
    lentot = len(data)
    sep = int(0.7*lentot)
    liste_ind = torch.randperm(lentot)
    batch_size = 16
    if cat == 'train':
        ind = liste_ind[:sep].tolist()
        data = Subset(data, ind)
        dataloader = DataLoader(data, batch_size= batch_size, shuffle = shuffle)
    else:
        ind = liste_ind[sep:].tolist()
        data = Subset(data, ind)
        dataloader = DataLoader(data, batch_size= batch_size, shuffle = shuffle)
    return dataloader 


class HandModeleMedium(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 5, padding = 0, stride = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, padding = 0, stride = 1)
        self.conv3 = nn.Conv2d(128, 512, kernel_size = 3, padding = 0, stride = 1)
        self.conv4 = nn.Conv2d(512, 2048, kernel_size = 3, padding = 0, stride = 1)
        self.conv5 = nn.Conv2d(2048, 2048, kernel_size = 3, padding = 0, stride = 1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(8192, 1024)
        self.linear2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
    
data_path = "D:/ProjetReachy/DataMain/PhotoMain"
output_csv = "D:/ProjetReachy/DataMain/label.csv"
noms = ['path', 'value']

with open(output_csv, mode = 'r', encoding = 'utf-8') as file:
    reader = csv.DictReader(file, fieldnames = noms)
    liste_data = list(map(conversion, reader))

transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

if __name__ == '__main__':
    #Entrainer le Modèle
    modele = HandModeleMedium().to(device)
    summary(modele, input_size = ((16, 3, 100, 100)))
    
    #préparation des données
    #Mettre en commun la fabication
    train_dataloader = former_dataset(liste_data, True, 'train')
    test_dataloader = former_dataset(liste_data, False, 'test')

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(modele.parameters(), lr = 1e-4)
    # PinceMain/train_check pour le Lite et PinceMain/train_check1 pour le HandModele1
    checkpoint_dir = 'PinceMain/train_check_medium.pth'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    EPOCH = 100

    if input('Do you want to load the model? (y/n)') == 'y':
        modele.load_state_dict(torch.load(checkpoint_dir))


    train(train_dataloader, EPOCH, modele, optimizer, criterion, checkpoint_dir)


    # Charger le modèle si nécessaire
    if input('Do you want to load the model for testing? (y/n)') == 'y':
        modele.load_state_dict(torch.load(checkpoint_dir))
        print("Modèle chargé.")

    # Tester le modèle avec affichage des différences
    if input('Do you want to test the model? (y/n)') == 'y':
        test_loss = test_model_with_accuracy(modele, test_dataloader, criterion, device)
