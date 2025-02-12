from torch.utils.data import DataLoader, Dataset
import soundfile as sf
import numpy as np
import torch
from torch import Tensor
from transformer.model_utils import seed_worker
import random
import fairseq

def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

def padding_zero(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    return np.concatenate((x, np.zeros(max_len-x_len)))
    
class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, vectorization_model):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.vectorization_model = vectorization_model

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index] # file names -> file name
        X, _ = sf.read(self.base_dir +'/'+ f"{key}.flac") # waveform
        X_pad = padding_zero(X, self.cut) # padding & tensorlize
        X_tensor = torch.from_numpy(X_pad).float().reshape(1, -1)
        
        res = self.vectorization_model.feature_extractor(X_tensor).squeeze(0).transpose(0,1) # audio
        y = self.labels[key] # label
        return res, y

def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train: # train
        for line in l_meta:
            _, key, _, cate, label = line.strip().split(" ")
            file_list.append(key)
            #repeat_num = random.randint(9, 18)
            repeat_num = 16
            generated_label = [7]
            if label == 'bonafide': res = 0
            else: res = int(cate[1:])
            for i in range(repeat_num): generated_label.append(res)
            generated_label.append(8)
            d_meta[key] = torch.tensor(generated_label)
            # bonafide: 0; spoofed: A01->1, A02->2, ..., sos->7, eos->8
            # [TODO] leverage one that could be used in Transformer. generate the number of labels reasonably. 
            
        return d_meta, file_list

    elif is_eval: # evaluate
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list

    else: # develop
        for line in l_meta:
            _, key, _, cate, label = line.strip().split(" ")
            file_list.append(key)
            #repeat_num = random.randint(9, 18)
            repeat_num = 16
            generated_label = [7]
            if label == 'bonafide': res = 0
            else: res = int(cate[1:])
            for i in range(repeat_num): generated_label.append(res)
            generated_label.append(8)
            d_meta[key] = torch.tensor(generated_label)
        return d_meta, file_list

def create_dataloader(audio_path, protocol_path, wav2vec2_path, seed=1234, bs=24):
    data_labels, file_list = genSpoof_list(protocol_path, is_train=True)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec2_path])
    model = model[0]
    model.eval()
    train_dataset = Dataset_ASVspoof2019_train(list_IDs=file_list, labels=data_labels, base_dir=audio_path, vectorization_model=model)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_dataset, 
                            batch_size=bs, 
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker, 
                            generator=gen)
    return trn_loader



if __name__ == "__main__":
    train_loader = create_dataloader(audio_path="D:/dataset/dataset/ASVSpoof 2019 LA/ASVspoof2019_LA_train/flac",
        protocol_path="D:/dataset/dataset/ASVSpoof 2019 LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.light.txt")