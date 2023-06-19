from dataset import get_datasets,get_data_loaders
from model import ClassifierModel
import torch
import config
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from utils import save_model,save_results,save_plots
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


dataset_train, dataset_valid, dataset_classes = get_datasets()
print(f"[INFO]: Number of training images: {len(dataset_train)}")
print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
print(f"[INFO]: Class names: {dataset_classes}\n")
# Load the training and validation data loaders.
train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)
# Learning parameters.


# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        preds = torch.argmax(probs.data, -1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, class_names):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    # We need two lists to keep track of class-wise accuracy.
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            # Forward pass.
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            preds = torch.argmax(probs.data, -1)

            valid_running_correct += (preds == labels).sum().item()
            # Calculate the accuracy for each class
            correct  = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    # # Print the accuracy for each class after every epoch.
    # print('\n')
    # for i in range(len(class_names)):
    #     print(f"Accuracy of class {class_names[i]}: {100*class_correct[i]/class_total[i]}")
    # print('\n')
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    model = ClassifierModel()

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    for epoch in range(config.EPOCHS):
        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                    criterion, dataset_classes)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)
        # Save the results.
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for data in tqdm(valid_loader, total=len(valid_loader)):
                images, labels = data
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())
        save_results(dataset_classes, true_labels, predicted_labels, 'outputs/results.json')
        # Calculate the confusion matrix.
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 8))
        classes = dataset_classes
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"outputs/confusion_matrix.png")
        plt.savefig(f"webapp/static/outputs/confusion_matrix.png")
        plt.close()

    # Save the trained model weights.
    save_model(config.EPOCHS, model, optimizer, criterion)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')
