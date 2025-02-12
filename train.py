import torch
import torch.optim as optim
from transformer.process import create_dataloader
from transformer.model_utils import set_seed
import json
from importlib import import_module
from typing import Dict
from utils import create_optimizer
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_config: Dict, device: torch.device):
    module = import_module("transformer.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model().to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask==0))
    return np_mask
def create_masks(src, trg):
    src_mask = None

    if trg is not None:
        trg_mask = trg.unsqueeze(-2).to(device)
        size = trg.size(1) 
        np_mask = nopeak_mask(size).to(device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

def train_epoch(dataloader, model, optimizer, scheduler, epoch_cur, epoch_nums):
    model.train()
    epoch_loss = 0
    step = 0
    for batch in tqdm(dataloader, desc=f'[{epoch_cur+1}/{epoch_nums}] training...'):
        audio = batch[0].to(device)
        label = batch[1].to(device)
        print(label)
        audio_mask, label_mask = create_masks(audio, label)
        if audio_mask != None: audio_mask.to(device)
        if label_mask != None: label_mask.to(device)
        
        output = model(audio, label, audio_mask, label_mask)
        reshape_label = label.view(-1)
        reshape_output = output.view(len(reshape_label), -1)
        loss = F.cross_entropy(reshape_output, reshape_label, ignore_index=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step += 1
        scheduler.step()
    return epoch_loss / step

def train(audio_path, protocol_path, vectorization_path, ckpt_path, seed = 1234, config_path = None):
    with open(config_path, "r") as f_json:
        config = json.loads(f_json.read())
    set_seed(seed, config)
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"
    

    dataloader = create_dataloader(audio_path, protocol_path, vectorization_path)
    model = get_model(model_config, device)
    optim_config["steps_per_epoch"] = len(dataloader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    best_running_loss = 1000
    for epoch in range(config["num_epochs"]):
        running_loss = train_epoch(dataloader, model, optimizer, scheduler, epoch, config["num_epochs"])
        print("loss: {}".format(running_loss))
        if running_loss < best_running_loss:
            best_running_loss = running_loss
            torch.save(model.state_dict(), os.path.join(ckpt_path, f"{epoch+1}_ckpt_loss_{running_loss}.pth"))



if __name__ == '__main__':
    train(audio_path="/mnt/data3/share/ASVSpoof 2019 LA/ASVspoof2019_LA_train/flac",
          protocol_path="/mnt/data3/share/ASVSpoof 2019 LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
          vectorization_path="/mnt/workspace/ganlinyong/fusion_model_test/wav2vec_small.pt", 
          config_path="/mnt/workspace/ganlinyong/transformer_audio_classification/configuration.conf", 
          ckpt_path = "/mnt/workspace/ganlinyong/transformer_audio_classification/ckpt")