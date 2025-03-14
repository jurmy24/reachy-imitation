"""Modèle de détection main ouverte ou fermé. Modèle le plus léger"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from torchvision import transforms
import csv
from HandGestureUtils import *
from PinceMain.HandGestureUtils import *


def former_dataset(liste_data, shuffle, cat):
    indices = torch.randperm(len(liste_data)).tolist()
    data = Subset(liste_data, indices)
    lentot = len(data)
    sep = int(0.7 * lentot)
    liste_ind = torch.randperm(lentot)
    batch_size = 16
    if cat == "train":
        ind = liste_ind[:sep].tolist()
        data = Subset(data, ind)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    else:
        ind = liste_ind[sep:].tolist()
        data = Subset(data, ind)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class HandModeleLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, padding=1, stride=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1, stride=1)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(9216, 1024)
        self.linear2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


data_path = "D:/ProjetReachy/DataMain/PhotoMain"
output_csv = "D:/ProjetReachy/DataMain/label.csv"
noms = ["path", "value"]

with open(output_csv, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file, fieldnames=noms)
    liste_data = list(map(conversion, reader))

transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

if __name__ == "__main__":
    modele = HandModeleLite().to(device)
    summary(modele, input_size=((128, 3, 100, 100)))

    # préparation des données
    train_dataloader = former_dataset(liste_data, True, "train")
    test_dataloader = former_dataset(liste_data, False, "test")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(modele.parameters(), lr=1e-4)
    # PinceMain/train_check pour le Lite et PinceMain/train_check1 pour le HandModele1
    checkpoint_dir = "PinceMain/train_check_lite.pth"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    EPOCH = 100

    if input("Do you want to load the model? (y/n)") == "y":
        modele.load_state_dict(torch.load(checkpoint_dir))

    train(train_dataloader, EPOCH, modele, optimizer, criterion, checkpoint_dir)

    # Charger le modèle si nécessaire
    if input("Do you want to load the model for testing? (y/n)") == "y":
        modele.load_state_dict(torch.load(checkpoint_dir))
        print("Modèle chargé.")

    # Tester le modèle avec affichage des différences
    if input("Do you want to test the model? (y/n)") == "y":
        test_loss = test_model_with_accuracy(modele, test_dataloader, criterion, device)
