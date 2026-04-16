import numpy as np
import scipy.io as sio

from torch.utils.data import Dataset
import torch

def load_timeseries(mat_dict, name):
    """Extract a Simulink Structure With Time saved in a .mat file."""
    obj = mat_dict[name][0, 0]
    t = np.asarray(obj["time"]).squeeze().astype(float)
    v = np.asarray(obj["signals"][0, 0]["values"]).squeeze().astype(float)
    return t, v


class InverseData(Dataset):
    def __init__(self, mat_file, outputs_npy, ks_npy):
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

        self.num_time_steps = len(self.output[0])

    def __len__(self):
        return len(self.output)
    
    def __getitem__(self, idx):
        output = self.output[idx]
        k = self.k[idx]

        model_input =torch.stack([
            torch.tensor(self.F_NOx_sensor[idx], dtype=torch.float32),
            torch.tensor(self.Dosing[idx], dtype=torch.float32),
            torch.tensor(self.Temp[idx], dtype=torch.float32),
            torch.tensor(self.ExhaustFlow[idx], dtype=torch.float32),
            torch.tensor(self.adblue_mg[idx], dtype=torch.float32),
            torch.tensor(self.O2[idx], dtype=torch.float32),
            torch.tensor(self.Temp_DOC_up[idx], dtype=torch.float32),
            torch.tensor(output, dtype=torch.float32)
        ], dim=0)
        # returns 8x11993 tensor 

        model_output = torch.tensor(k, dtype=torch.float32)
        # return 1x6 tensor


        ##### process the above tensors as needed for your model #####

        random_time = torch.randint(0, self.num_time_steps)



        return model_input, model_output

