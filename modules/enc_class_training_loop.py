import torch
import os
import pandas as pd
import json
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# modules directory
from encoder_classifier import ConvClassifier
import utils

def train_and_save_conv_classifier(config, num_epochs, save_filepath, dataset, batch_size=64):
    """
    Trains a convolutional autoencoder model of given parameters with:
        initial lr = 0.01
        optimizer = adam
        lr scheduler = steplr or reduce on plateau or cosine annealing
    until loss plateaus again or validation loss increaaases (determined by training heuristics)
    """
    
    ldim=config['latent_dim']
    hdim=config['hidden_dim']
    conv_blocks=config['conv_blocks']
    k=config['kernel']
    id=config['id']

    # --------------- LOAD DATASET -----------------

    # retrieve 1-second frames from csv 
    b_size = batch_size
    n_workers = utils.optimal_num_workers()

    frames = dataset

    # split 'frames' dataset into train, val, and test
    g = torch.Generator().manual_seed(7)

    frames_trainset, frames_valset, frames_testset = random_split(frames, [0.8, 0.1, 0.1], generator=g)

    frames_trainloader = DataLoader(frames_trainset, batch_size=b_size, num_workers=n_workers,pin_memory=torch.cuda.is_available(),persistent_workers=True, prefetch_factor=4)
    frames_valloader = DataLoader(frames_valset, batch_size=b_size, num_workers=n_workers,pin_memory=torch.cuda.is_available(),persistent_workers=True, prefetch_factor=4)
    frames_testloader = DataLoader(frames_testset, batch_size=b_size, num_workers=n_workers,pin_memory=torch.cuda.is_available(),persistent_workers=True, prefetch_factor=4)

    input_shape = tuple(frames_trainset.__getitem__(0)[0].shape[-2:])

    # ---------- LOAD MODEL ARCHITECTURE -------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier = ConvClassifier(input_shape=input_shape, latent_dim=ldim, channels=conv_blocks, hidden_dim=hdim, kernel=k, num_classes=4)
    classifier.to(device)

    criterion = nn.CrossEntropyLoss()

    # test model architecture with forward pass
    try:
        with torch.no_grad():
            # unsqueeze adds batch dimension to (C, W, H) and (num_classes) tensors
            sample_x, sample_y = next(iter(frames_trainloader))[0].to(device), next(iter(frames_trainloader))[1].to(device)
            sample_classification = classifier(sample_x)

            # compares predicted and ground truth distributions
            loss = criterion(sample_classification, sample_y.float())

            print(f'Modell initialised. Shapes: \n {sample_x.shape} {sample_y.shape} {loss.item()}')

    except Exception as e:
        print(f'Exception occured: \n {e} \n Possibly caused by invalid parameters')


    # ------------ HANDLE METADATA ----------------------

    # model_name is used to save config, weights, and training statistics

    model_filepath = save_filepath

    model_name = f'conv_classifier_ldim{ldim}_convblocks{len(conv_blocks)}_hdim{hdim}_kernel{k}_id{id}'

    # create a tensorboard writer object that logs to /tensorboard_logs subdir
    writer = SummaryWriter(log_dir=os.path.join(model_filepath, "tensorboard_logs", model_name))


    # save config file
    config['parameters'] = utils.num_parameters(classifier)
    with open(model_filepath+model_name+'.json', 'w') as file:
        json.dump(config, file, indent=4)

    # check for existing state dicts
    if f'{model_name}.pth' in os.listdir(model_filepath):
        classifier.load_state_dict(torch.load(model_filepath+model_name+'.pth'))
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

    # determine global step for tensorboard logs (total number of epochs across all runs)
    if model_name+'.csv' in os.listdir(model_filepath):
        global_step = len(pd.read_csv(os.path.join(model_filepath, model_name+'.csv')))
    else:
        global_step = 0

    print(f'Ready to begin training from {global_step} epoch. \n {device, model_filepath, model_name}')

    # ------------ TRAINING PARAMETERS -----------------

    epochs = num_epochs
    patience = 20
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    optimizer = optim.Adam(classifier.parameters(), lr=1e-2)

    reduce_on_plateau = ReduceLROnPlateau(optimizer, cooldown=10, min_lr=1e-5)
    # step_lr = StepLR(optimizer, step_size=40, gamma=0.1)
    # cos_lr = CosineAnnealingLR(optimizer, T_max=epochs//10, eta_min=0)

    scheduler = reduce_on_plateau

    # ---------------- TRAINING LOOP -------------------

    for e in range(epochs):

        # log data
        embeddings = []
        labels = []

        classifier.train()
        train_loss = 0.0
        for data, target in tqdm(frames_trainloader, desc=f"Epoch {e+1}/{epochs} [Training]", leave=False):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            encoded = classifier.encode(data)
            classification = classifier.classify(encoded)
            loss = criterion(classification, target.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            writer.add_scalar("Loss/Train", loss.item(), global_step+e)

            if e % 50 == 0:
                # (b_size, embed_dim) for each minibatch
                embeddings.append(encoded)
                # (b_size, num_classes) for each minibatch
                labels.append(target)

        scheduler.step(train_loss)

        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_step+e)
        
        # log activation data periodically
        if e % 5 == 0:
            for name, param in classifier.named_parameters():
                if param.grad is not None:

                    # log parameter update to parameter magnitude ratio
                    param_norm = torch.norm(param).item()
                    update_norm = torch.norm(param.grad * optimizer.param_groups[0]['lr']).item()
                    ratio = update_norm / (param_norm + 1e-10) 
                    writer.add_scalar(f"Update-to-Param/{name}", ratio, global_step+e)

                    # weight and gradient distributions
                    # writer.add_histogram(f"Weights/{name}", param, global_step+e)
                    # writer.add_histogram(f"Gradients/{name}", param.grad, global_step+e)

        # save embeddings for t-SNE visualization
        if e % 50 == 0:
            writer.add_embedding(
                # concatentate each minibatche's 2d tensor
                torch.cat(embeddings, dim=0),
                metadata=torch.argmax(torch.cat(labels, dim=0), dim=1),
                tag=f"Embeddings_epoch_{e}",
                global_step=global_step+e
            )

        classifier.eval()
        val_loss = 0.0
        truepos = 0.0
        with torch.no_grad():
            for data, target in tqdm(frames_valloader, desc=f"Epoch {e+1}/{epochs} [Validation]", leave=False):
                data = data.to(device)
                target = target.to(device)
                classification = classifier(data)
                loss = criterion(classification, target.float())

                if torch.argmax(classification) == torch.argmax(target):
                    truepos += 1

                val_loss += loss.item()
                writer.add_scalar("Loss/Validation", loss.item(), global_step+e)
    

        # average losses
        train_loss /= len(frames_trainloader)
        val_loss /= len(frames_valloader)
        accuracy = truepos / len(frames_valloader)

        #print(f"Epoch {e+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        log.append({'Epoch':e+1, 'Training Loss':round(train_loss, 4), 'Validation Loss':round(val_loss, 4), 'Accuracy':accuracy, 'Learning Rate':scheduler.get_last_lr()[0]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # if epochs_without_improvement >= patience:
        #     epochs_without_improvement = 0
        #     best_val_loss = 0.0
        #     print(f'Plateaued at {e+1}, switching to cosine annealing ')
        #     scheduler = cos_lr

        if e % 5 == 0:
            torch.save(classifier.state_dict(), model_filepath + model_name + '.pth')
            loss_data = pd.DataFrame(log)
            loss_data.to_csv(model_filepath+model_name+'.csv', index=False)

            writer.flush()

            # past_train_loss = loss_data['Training Loss'][-patience:].values
            # past_val_loss = loss_data['Validation Loss'][-patience:].values
            # local_x = np.arange(len(past_train_loss))

            # train_grad = np.gradient(past_train_loss, local_x)
            # val_grad = np.gradient(past_val_loss, local_x)

            # if np.mean(train_grad)/np.mean(val_grad) > 2:
            #     print(f'Potential overfitting detected at epoch {e}. Breaking off training')
            #     break

    # save model weights 
    torch.save(classifier.state_dict(), model_filepath + model_name + '.pth')
    loss_data = pd.DataFrame(log)
    loss_data.to_csv(model_filepath+model_name+'.csv', index=False)

    writer.flush()
    writer.close()

    print(f'Model trained for {e} epochs. Saved data for {model_name}')

    return accuracy

if  __name__ == '__main__':

    # run different configs
    # tensor from dataloader has shape (frame_length, num_channels). Padding applied
    configs = [
        {'latent_dim': 32, 'conv_blocks': [1, 16], 'hidden_dim':16, 'kernel': (3, 2), 'id': 'TEST'}
    ]

    save_filepath = "saved_models\\classifier\\TESTS\\"
    training_epochs = 1

    for config in configs:
        train_and_save_conv_classifier(config=config, num_epochs=training_epochs, save_filepath=save_filepath)

    # then run: %tensorboard --logdir=os.path.join(model_filepath, "tensorboard_logs", model_name)
