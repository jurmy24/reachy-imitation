"""Outils pour construction et entrainement des réseaux neuronnaux"""

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda")
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()


def conversion(dic):
    return [dic["path"], dic["value"]]


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print("modele enregistré")


transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])


def tensorisation(file_paths):
    image_tensor = []
    for path in file_paths:
        image = Image.open(path)
        tensor = transform(image)
        image_tensor.append(tensor)
    return torch.stack(image_tensor)


def process_label(labels):
    return torch.tensor(list(map(float, labels))).unsqueeze(1)


def train_step(batch, modele, optimizer, criterion):
    optimizer.zero_grad()
    X, label = batch
    label_pred = modele(tensorisation(X).to(device))
    loss = criterion(label_pred, process_label(label).to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


def train(data, EPOCH, modele, optimizer, criterion, save_path):
    for epoch in range(1, EPOCH + 1):
        print("\n Epoch {}/{}".format(epoch, EPOCH))
        running_loss = 0.0
        for idx, batch in enumerate(data):
            running_loss += train_step(batch, modele, optimizer, criterion)
        if epoch % 10 == 0:
            save_checkpoint(modele, save_path)
        print(f"la loss de l' epoch {epoch} vaut {running_loss}")


def test_model_with_accuracy(model, test_dataloader, criterion, device):
    model.eval()  # Mode évaluation
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Pas de calcul de gradients pendant le test
        for batch_idx, batch in enumerate(test_dataloader):
            print(batch_idx)
            X, labels = batch

            # Prétraitement des données
            X = tensorisation(X).to(device)
            labels = process_label(labels).to(device)

            # Prédictions
            predictions = model(X)
            # print(predictions)
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            # Conversion des prédictions en 0 ou 1
            predicted_classes = (predictions >= 0.5).int()  # Seuil à 0.5
            correct_predictions = (predicted_classes == labels).sum().item()

            # Comptage des échantillons et des prédictions correctes
            total_correct += correct_predictions
            total_samples += labels.size(0)

            # Affichage des résultats pour le batch courant
            print(f"\nBatch {batch_idx + 1}/{len(test_dataloader)}")
            print(f"Prédictions Correctes : {correct_predictions}/{labels.size(0)}")

    # Moyenne des pertes et précision totale
    avg_loss = total_loss / len(test_dataloader)
    accuracy = total_correct / total_samples
    print(f"\nTest Loss Moyenne: {avg_loss:.4f}")
    print(f"Précision Totale: {accuracy:.4%}")
    return avg_loss, accuracy


def affiche_image(img):
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.show()
