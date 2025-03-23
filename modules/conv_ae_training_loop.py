import torch
import os
import pandas as pd
import json

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

import csv
from tqdm import tqdm

# modules directory
from frame_dataloader_heavy import WorkloadFrame
from convolutional_autoencoder import ConvAE
import utils


def train_and_save_conv_ae(config, num_epochs, save_filepath):
    """
    Trains a convolutional autoencoder model of given parameters with:
        initial lr = 0.01
        optimizer = adam
        lr scheduler = steplr
    until loss decrease plateaus, at which point it switches to 
        cosine annealing scheduling 
    until loss plateaus again or validation loss increaaases
    """
    
    dim=config['latent_dim']
    conv_blocks=config['conv_blocks']
    k=config['kernel']
    id=config['id']

    # --------------- LOAD DATASET -----------------

    # retrieve 1-second frames from csv 
    b_size = 64
    n_workers = utils.optimal_num_workers()

    # directory to .mat files
    if __name__ == '__main__':
        dir = 'files'
    else:
        dir='..\\files'

    #  file group: 'phys', 'cog', or 'tot'
    group='phys'

    # signal channel to resample to: 'temp', 'hrv, 'hr', 'hbo', 'eda'
    resample='temp'

    # size of sliding window relative to shortest signal length; always 50% overlap between windows
    context_length=0.5

    frames = WorkloadFrame(dir=dir, group=group, resample=resample, context_length=context_length)

    # split 'frames' dataset into train, val, and test
    g = torch.Generator().manual_seed(7)

    frames_trainset, frames_valset, frames_testset = random_split(frames, [0.8, 0.1, 0.1], generator=g)

    frames_trainloader = DataLoader(frames_trainset, batch_size=b_size, num_workers=n_workers,pin_memory=torch.cuda.is_available(),persistent_workers=True, prefetch_factor=4)
    frames_valloader = DataLoader(frames_valset, batch_size=b_size, num_workers=n_workers,pin_memory=torch.cuda.is_available(),persistent_workers=True, prefetch_factor=4)
    frames_testloader = DataLoader(frames_testset, batch_size=b_size, num_workers=n_workers,pin_memory=torch.cuda.is_available(),persistent_workers=True, prefetch_factor=4)

    # ---------- LOAD MODEL ARCHITECTURE -------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conv_ae = ConvAE(latent_dim=dim, channels=conv_blocks, kernel=k)
    conv_ae.to(device)

    # initialise and test model architecture with forward pass
    try:
        with torch.no_grad():
            sample_input = frames_trainset.__getitem__(0)[0].unsqueeze(0).to(device)
            sample_output = conv_ae(sample_input)
            loss = F.mse_loss(sample_input, sample_output, reduction='mean')
            print(f'Modell initialised. Shapes: \n {sample_input.shape} {sample_output.shape} {loss.item()}')

    except Exception as e:
        print(f'Exception occured: \n {e} \n Possibly caused by invalid parameters')

    def loss_function(y, y_hat, red='mean'):
        return F.mse_loss(y, y_hat, reduction=red)

    # ------------ HANDLE METADATA ----------------------

    # model_name is used to save config, weights, and training statistics

    model_filepath = save_filepath

    model_name = f'conv_ae_ndim{dim}_convblocks{len(conv_blocks)}_kernel{k}_id{id}'

    # save config file
    with open(model_filepath+model_name+'.json', 'w') as file:
        json.dump(config, file, indent=4)

    # check for existing state dicts
    if f'{model_name}.pth' in os.listdir(model_filepath):
        conv_ae.load_state_dict(torch.load(model_filepath+model_name+'.pth'))
        print(f'Loaded state dict for {model_name}')
    else:
        print(f'No saved weights found for {model_name} in {model_filepath}')

    # check for existing logs
    if f'{model_name}.csv' in os.listdir(model_filepath):
        df = pd.read_csv(model_filepath+model_name+'.csv')
        log = list(df.T.to_dict().values())
        print(f'Existing log data found for {len(log)} epochs')
    else:
        log = []
        print(f'No existing log data found for {model_name} in {model_filepath}. Creating new one')

    # activation statistics
    activation_stats = {}

    def forward_hook(module, input, output):
        if conv_ae.log_stats:
            activation_stats[module] = {
                "mean": output.mean().item(),
                "std": output.std().item()
        }

    for name, layer in conv_ae.named_modules(remove_duplicate=True):
        layer.register_forward_hook(forward_hook)

    print(f'Ready to begin training {device, model_filepath, model_name}')

    # ------------ TRAINING PARAMETERS -----------------

    epochs = num_epochs
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    optimizer = optim.Adam(conv_ae.parameters(), lr=1e-3)

    step_lr = StepLR(optimizer, step_size=50, gamma=0.1)
    cos_lr = CosineAnnealingLR(optimizer, T_max=epochs//10, eta_min=0)

    scheduler = step_lr

    # ---------------- TRAINING LOOP -------------------

    for e in range(epochs):
        conv_ae.train()
        train_loss = 0.0
        for data, _ in tqdm(frames_trainloader, desc=f"Epoch {e+1}/{epochs} [Training]", leave=False):
            data = data.to(device)

            optimizer.zero_grad()
            recon_data = conv_ae(data)
            loss = loss_function(recon_data, data, red='mean')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        scheduler.step()

        conv_ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in tqdm(frames_trainloader, desc=f"Epoch {e+1}/{epochs} [Validation]", leave=False):
                data = data.to(device)
                recon_data = conv_ae(data)
                loss = loss_function(recon_data, data, red='mean')

                val_loss += loss.item()

        # average losses
        train_loss /= len(frames_trainloader)
        val_loss /= len(frames_trainloader)

        print(f"Epoch {e+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        log.append({'Epoch':e+1, 'Training Loss':round(train_loss, 4), 'Validation Loss':round(val_loss, 4), 'Learning Rate':scheduler.get_last_lr()[0]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            epochs_without_improvement = 0
            best_val_loss = 0.0
            print(f'Plateaued at {e+1}, switching to cosine annealing ')
            scheduler = cos_lr

        if e % 5 == 0:
            torch.save(conv_ae.state_dict(), model_filepath + model_name + '.pth')
            loss_data = pd.DataFrame(log)
            loss_data.to_csv(model_filepath+model_name+'.csv', index=False)

    # save model weights and training statistics
    torch.save(conv_ae.state_dict(), model_filepath + model_name + '.pth')
    loss_data = pd.DataFrame(log)
    loss_data.to_csv(model_filepath+model_name+'.csv', index=False)

    # save activation stats
    sample_input = frames_trainset.__getitem__(0)[0].unsqueeze(0).to(device)

    # set log_stats flag to true to log activation statistics for each module
    conv_ae.log_stats = True
    with torch.no_grad():
        sample_output = conv_ae(sample_input)
    conv_ae.log_stats = False
    s = []
    m = []
    for module, stats in activation_stats.items():
        s.append(stats['std'])
        m.append(stats['mean'])

    activation_map = {}
    activation_map['std'] = s
    activation_map['mean'] = m

    with open(model_filepath+model_name+'_activations.json', 'w') as file:
        json.dump(activation_map, file, indent=4)

    print(f'Model trained for {e} epochs. Saved data for {model_name}')

if  __name__ == '__main__':

    # run different configs
    configs = [
        {'latent_dim': 1024, 'conv_blocks': [1, 64, 128], 'kernel': (3, 2), 'id': None}
    ]

    save_filepath = "saved_models\\convolutional_autoencoder\\"
    training_epochs = 1

    for config in configs:
        train_and_save_conv_ae(config=config, num_epochs=training_epochs, save_filepath=save_filepath)
