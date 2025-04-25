import torch
import os
import pandas as pd
import json
import numpy as np
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# modules directory
import utils
from composite_model import CompModel
from wrapper_class import ModelHead

def train_and_save_comp_model(config: dict, model_list: list, dataset, save_filepath: str, id, step_size=10, gamma=0.1):
    """
    Trains a composite model of given parameters with:
        optimizer = adam
        lr scheduler = steplr or reduce on plateau or cosine annealing
    until loss plateaus again or validation loss increaaases (determined by training heuristics)
    """


    # --------------- LOAD DATASET -----------------

    # retrieve 1-second frames from csv 
    b_size = config['batch_size']
    n_workers = 0 if 'ipykernel' in sys.modules else utils.optimal_num_workers()
    p_workers = ('ipykernel' not in sys.modules)

    # split 'frames' dataset into train, val, and test
    g = torch.Generator().manual_seed(7)

    frames_trainset, frames_valset, frames_testset = random_split(dataset, [0.8, 0.1, 0.1], generator=g)

    frames_trainloader = DataLoader(frames_trainset, batch_size=b_size, pin_memory=torch.cuda.is_available(),persistent_workers=p_workers, num_workers=n_workers)
    frames_valloader = DataLoader(frames_valset, batch_size=b_size, pin_memory=torch.cuda.is_available(), persistent_workers=p_workers, num_workers=n_workers)
    frames_testloader = DataLoader(frames_testset, batch_size=b_size, pin_memory=torch.cuda.is_available(),persistent_workers=p_workers, num_workers=n_workers)

    input_shape = tuple(frames_trainset.__getitem__(0)[0].shape[-2:])

    if len(frames_trainloader) == 0 or len(frames_valloader) == 0:
        raise ValueError("DataLoader is empty! Check dataset loading.")

    # ---------- LOAD MODEL ARCHITECTURE -------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_dim = config['hidden_dim']
    model_heads = [ModelHead(**m) for m in model_list]

    classifier = CompModel(model_heads, hidden_dim, latent_dim=None, num_classes=4)
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

    model_name = f'comp_classifier_{id}_hdim{hidden_dim}_param{utils.num_parameters(classifier)}'
    
    if os.path.exists(model_filepath) and any([(model_name in f) for f in os.listdir(model_filepath)]):
        raise Exception('Model name exists in directory! Choose another ID')
    

    # NOTE: logging dir must be created first
    # create a tensorboard writer object that logs to /tensorboard_logs subdir
    writer = SummaryWriter(log_dir=os.path.join(model_filepath, "tensorboard_logs", model_name))
    # save config file
    with open(model_filepath+model_name+'.json', 'w') as file:
        json.dump(config, file, indent=4)


    # check for existing logs
    # determine global step for tensorboard logs (total number of epochs across all runs)
    if os.path.exists(model_filepath) and (f'{model_name}.csv' in os.listdir(model_filepath)):
        df = pd.read_csv(model_filepath+model_name+'.csv')
        global_step = len(df)
        log = list(df.T.to_dict().values())
        print(f'Existing log data found for {len(log)} epochs')
    else:
        log = []
        global_step=0
        print(f'No existing log data found for {model_name} in {model_filepath}. Creating new one')


    print(f'Ready to begin training from {global_step} epoch. \n {device, model_filepath, model_name}')

    # ------------ TRAINING PARAMETERS -----------------

    epochs = config['num_epochs']
    optimizer = optim.Adam(classifier.parameters(), lr=config['initlr'])

    #reduce_on_plateau = ReduceLROnPlateau(optimizer, cooldown=10, min_lr=1e-5)
    step_lr = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # cos_lr = CosineAnnealingLR(optimizer, T_max=epochs//10, eta_min=0)

    scheduler = step_lr

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

            if e % (epochs-1) == 0:
                # (b_size, embed_dim) for each minibatch
                embeddings.append(encoded)
                # (b_size, num_classes) for each minibatch
                labels.append(target)

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
        if e % (epochs-1) == 0:
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
        total_evals = 0.0
        with torch.no_grad():
            for data, target in tqdm(frames_valloader, desc=f"Epoch {e+1}/{epochs} [Validation]", leave=False):
                data = data.to(device)
                target = target.to(device)
                classification = classifier(data)
                loss = criterion(classification, target.float())

                truepos += torch.sum(torch.argmax(F.softmax(classification, dim=1), dim=1) == torch.argmax(target, dim=1))
                total_evals += classification.shape[0]

                val_loss += loss.item()
                writer.add_scalar("Loss/Validation", loss.item(), global_step+e)
    

        # average losses
        train_loss /= len(frames_trainloader)
        val_loss /= len(frames_valloader)
        accuracy = float(truepos / total_evals)

        #print(f'ACCURACY: {accuracy} TRUEPOS {truepos} EVALS {total_evals}')

        scheduler.step()

        #print(f"Epoch {e+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        log.append({'Epoch':e+1, 'Training Loss':round(train_loss, 4), 'Validation Loss':round(val_loss, 4), 'Accuracy':accuracy, 'Learning Rate':scheduler.get_last_lr()[0]})


        if e % 5 == 0:
            torch.save(classifier.state_dict(), model_filepath + model_name + '.pth')
            loss_data = pd.DataFrame(log)
            loss_data.to_csv(model_filepath+model_name+'.csv', index=False)

            for i, h_w in enumerate(model_heads):
                torch.save(h_w.state_dict(), model_filepath + f"head{i}.pth")

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

    # print(f'Model trained for {e} epochs. Saved data for {model_name}')
    # print(f"DEBUG: truepos={truepos}, total_evals={total_evals}, accuracy={accuracy}")
    # print(f"DEBUG: train={len(frames_trainloader)}, val={len(frames_valloader)}, test={len(frames_testloader)}")
    return accuracy
