import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time
from tqdm import tqdm

def load_timeseries(mat_dict, name):
    obj = mat_dict[name][0, 0]
    t = np.asarray(obj["time"]).squeeze().astype(float)
    v = np.asarray(obj["signals"][0, 0]["values"]).squeeze().astype(float)
    return t, v

class InverseData(Dataset):
    def __init__(self, mat_file, outputs_npy, ks_npy, save_norm=False):
        mat = sio.loadmat(mat_file)

        self.t, self.F_NOx_sensor = load_timeseries(mat, "F_NOx_sensor")
        _, self.Dosing = load_timeseries(mat, "Dosing")
        _, self.Temp = load_timeseries(mat, "Temp")
        _, self.ExhaustFlow = load_timeseries(mat, "ExhaustFlow")
        _, self.adblue_mg = load_timeseries(mat, "adblue_mg")
        _, self.O2 = load_timeseries(mat, "O2")
        _, self.Temp_DOC_up = load_timeseries(mat, "Temp_DOC_up")

        self.output = np.load(outputs_npy)
        self.k = np.load(ks_npy)
        # self.k = np.log10(self.k)
        
        self.num_time_steps = len(self.output[0])

        self.modelInput_NORM_PARAMS = {
            "F_NOx_sensor" : {"index" : 0, "max" : max(self.F_NOx_sensor), "min" : min(self.F_NOx_sensor)},
            "Dosing" : {"index" : 1, "max" : max(self.Dosing), "min" : min(self.Dosing)},
            "Temp" : {"index" : 2, "max" : max(self.Temp), "min" : min(self.Temp)},
            "ExhaustFlow" : {"index" : 3, "max" : max(self.ExhaustFlow), "min" : min(self.ExhaustFlow)},
            "adblue_mg" : {"index" : 4, "max" : max(self.adblue_mg), "min" : min(self.adblue_mg)},
            "O2" : {"index" : 5, "max" : max(self.O2), "min" : min(self.O2)},
            "Temp_DOC_up" : {"index" : 6, "max" : max(self.Temp_DOC_up), "min" : min(self.Temp_DOC_up)},
            "output" : {"index" : 7, "max" : np.max(self.output), "min" : np.min(self.output)}
        }

        self.modelOutput_NORM_PARAMS = {
            "k_ads" : {"index" : 0, "mean" : np.mean(self.k[:,0]), "std" : np.std(self.k[:,0])},
            "k_des" : {"index" : 1, "mean" : np.mean(self.k[:,1]), "std" : np.std(self.k[:,1])},
            "k_std" : {"index" : 2, "mean" : np.mean(self.k[:,2]), "std" : np.std(self.k[:,2])},
            "k_fst" : {"index" : 3, "mean" : np.mean(self.k[:,3]), "std" : np.std(self.k[:,3])},
            "k_slw" : {"index" : 4, "mean" : np.mean(self.k[:,4]), "std" : np.std(self.k[:,4])},
            "k_oxi" : {"index" : 5, "mean" : np.mean(self.k[:,5]), "std" : np.std(self.k[:,5])},
        }

        if save_norm:
            with open("./norm_params/input_norm.json", "w") as f:
                json.dump(self.modelInput_NORM_PARAMS, f)
            
            with open("./norm_params/output_norm.json", "w") as f:
                json.dump(self.modelOutput_NORM_PARAMS, f)

        self.F_NOx_sensor = (self.F_NOx_sensor - min(self.F_NOx_sensor))/(max(self.F_NOx_sensor) - min(self.F_NOx_sensor))
        self.Dosing= (self.Dosing - min(self.Dosing))/(max(self.Dosing) - min(self.Dosing))
        self.Temp = (self.Temp - min(self.Temp))/(max(self.Temp) - min(self.Temp))
        self.ExhaustFlow = (self.ExhaustFlow - min(self.ExhaustFlow))/(max(self.ExhaustFlow) - min(self.ExhaustFlow))
        self.adblue_mg = (self.adblue_mg - min(self.adblue_mg))/(max(self.adblue_mg) - min(self.adblue_mg))
        self.O2 = (self.O2 - min(self.O2))/(max(self.O2) - min(self.O2))
        self.Temp_DOC_up = (self.Temp_DOC_up - min(self.Temp_DOC_up))/(max(self.Temp_DOC_up) - min(self.Temp_DOC_up))
        self.output = (self.output - np.min(self.output))/(np.max(self.output) - np.min(self.output))

        self.k[:,0] = (self.k[:,0] - np.mean(self.k[:,0])) / np.std(self.k[:,0])
        self.k[:,1] = (self.k[:,1] - np.mean(self.k[:,1])) / np.std(self.k[:,1])
        self.k[:,2] = (self.k[:,2] - np.mean(self.k[:,2])) / np.std(self.k[:,2])
        self.k[:,3] = (self.k[:,3] - np.mean(self.k[:,3])) / np.std(self.k[:,3])
        self.k[:,4] = (self.k[:,4] - np.mean(self.k[:,4])) / np.std(self.k[:,4])
        self.k[:,5] = (self.k[:,5] - np.mean(self.k[:,5])) / np.std(self.k[:,5])

        self.trunk_input_base = torch.stack([
            torch.tensor(self.F_NOx_sensor, dtype=torch.float32),
            torch.tensor(self.Dosing, dtype=torch.float32),
            torch.tensor(self.Temp, dtype=torch.float32),
            torch.tensor(self.ExhaustFlow, dtype=torch.float32),
            torch.tensor(self.adblue_mg, dtype=torch.float32),
            torch.tensor(self.O2, dtype=torch.float32),
            torch.tensor(self.Temp_DOC_up, dtype=torch.float32)
        ], dim=0).permute(1, 0)

    def __len__(self):
        return len(self.output)
    
    def __getitem__(self, idx):
        output = torch.tensor(self.output[idx], dtype=torch.float32)
        branch_input = torch.tensor(self.k[idx], dtype=torch.float32)
        trunk_input = self.trunk_input_base

        return branch_input, trunk_input, output
