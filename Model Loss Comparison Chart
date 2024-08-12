# This performance metric should be used to compare performance of model hyperparameter and architecture 
# experiments across many epochs.

# NOTE: Program was built under the assumption that the training folder and experiment folder are in the
# same local directory as this file

# Dependencies
import matplotlib.pyplot as plt
import torch
import re
from natsort import natsorted
import os
import shutil


# MAKE SURE TO CHANGE EXPERIMENT NAME TO MOST RECENT BEFORE RUNNING OR IT WILL OVERWRITE
# Choose experiment title (must include 'experiment' in title)
experiment_name = 'experiment 1 - 360x4 layers'
# Name of folder where necessary files and experiment figures will be saved
experiments_dir = ''
# Name of folder containing most recent training files
recent_training_folder_path = ''

# Input necessary information about model training session
num_epochs = 10000
checkpoint = 1000

# Define loss metric
loss_function=torch.nn.MSELoss(reduction='mean')

# Load validation sets to test models:
input_torch = torch.load('path to input data')
label_torch = torch.load('path to label data')


# Create new directory to save imported files of recent experiment
os.makedirs(os.path.join(experiments_dir, experiment_name), exist_ok=True)
specific_experiment_dir = os.path.join(experiments_dir, experiment_name)


# Get the correct saves from the recently trained model and copy over
# NOTE: This was built for model saves similar to this: 'model_epoch_20.pt'
for save in sorted(os.listdir(recent_training_folder_path)): 
    if 'model_epoch_' in save:
        for i in range(0, num_epochs, checkpoint):
            # if current save is one of the correct numbers supposed to be saved, save it
            if int(re.findall(r'[\d]+', save)[0]) == int(i):
                 shutil.copy(os.path.join(recent_training_folder_path, save), specific_experiment_dir)


# Adjust figure size for readability
plt.figure(figsize=(10,10))

# Generate loss graph for every experiment
for i in os.listdir(experiments_dir):
    epochs_list=[]
    loss_epochs=[]
    if 'experiment' in i:
        specific_model_saves_path = os.path.join(experiments_dir, i)
        for save in natsorted((os.listdir(specific_model_saves_path))):
            current_model = torch.load(os.path.join(specific_model_saves_path, save))
            loss = loss_function(current_model(input_torch), label_torch)
            epochs_list.append(int(re.findall(r'[\d]+', save)[0]))
            loss_epochs.append(loss.item())
        plt.plot(epochs_list, loss_epochs, label=i)
        
# Set up plot and save figure to experiments folder
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Experiments")
plt.legend()
plt.savefig(os.path.join(experiments_dir, 'Comparison'))
plt.show()
