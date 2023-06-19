import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt


matplotlib.style.use('ggplot')
def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"outputs/model.pth")



def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    epochs = list(range(len(train_acc)))

    plt.plot(epochs, train_acc, color='green', linestyle='-', label='train accuracy')
    plt.plot(epochs, valid_acc, color='blue', linestyle='-', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("outputs/accuracy.png")
    plt.savefig("webapp/static/outputs/accuracy.png")

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_loss, color='orange', linestyle='-', label='train loss')
    plt.plot(epochs, valid_loss, color='red', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("outputs/loss.png")
    plt.savefig("webapp/static/outputs/loss.png")
    


def save_results(class_names, true_labels, predicted_labels, filepath):
    cm = confusion_matrix(true_labels, predicted_labels)
    accuracy, recall, precision, f2_score = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted', labels=np.unique(predicted_labels))
    results = {
        'class_names': class_names,
        'confusion_matrix': cm.tolist(),
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        # 'f2_score': f2_score
    }
    with open(filepath, 'w') as file:
        json.dump(results, file)

    with open("webapp/static/outputs/results.json","w") as file:
        json.dump(results,file)
    