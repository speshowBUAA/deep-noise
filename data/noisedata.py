import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import math

class NoiseData(Dataset):
    def __init__(self, dir='../data', filename='data_final_train.xlsx', use_type=None, transform=None):
        self.dir = dir
        self.filename = filename
        self.use_type = use_type
        self.transform = transform
        try:
            if 'train' in filename:
                all_sheets = pd.read_excel(os.path.join(self.dir, 'data_final_train.xlsx'), sheet_name=None)
            elif 'test' in filename:
                all_sheets = pd.read_excel(os.path.join(self.dir, 'data_final_test.xlsx'), sheet_name=None)
            else:
                all_sheets = pd.read_excel(os.path.join(self.dir, self.filename), sheet_name=None)
        except Exception as e:
            raise e
        self.dataFrame = pd.DataFrame()
        self.sheet_names = []
        self.sheet_indices = []
        
        for sheet_idx, (sheet_name, df) in enumerate(all_sheets.items()):
            self.sheet_names.append(sheet_name)
            df['sheet_idx'] = sheet_idx
            self.dataFrame = pd.concat([self.dataFrame, df], ignore_index=True)
        
        self.le = []
        keys = self.dataFrame.keys()
        if self.use_type:
            le = LabelEncoder()
            self.dataFrame[keys[0]] = le.fit_transform(self.dataFrame[keys[0]])
            # print(le.fit_transform(['AD02','BYD HTH','E0Y-3Z3M','GEM','H37','H97D','KKL','M1E','MAR2 2Z','MAR2 EVA2','MBQ','MEB','NU2','SA5H','SRH','T1X','T2X RHD','X03']))
            self.le = le.classes_

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        keys = self.dataFrame.keys()
        input = [self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx]]
        output = [self.dataFrame[keys[len(keys)-1]][idx]]
        if self.transform is not None:
            input = self.transform(input)
        input = torch.tensor(input).to(torch.float32)
        output = torch.tensor(output).to(torch.float32)
        if self.use_type:
            input = [self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx]]
            output = [self.dataFrame[keys[len(keys)-2]][idx]]
            type = [self.dataFrame[keys[0]][idx]]
            sheet_idx = self.dataFrame['sheet_idx'][idx]
        else:
            input = [self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx]]
            output = [self.dataFrame[keys[len(keys)-2]][idx]]
            type = 0
            sheet_idx = self.dataFrame['sheet_idx'][idx]

        if self.transform is not None:
            input = self.transform(input)
        
        if isinstance(input, torch.Tensor):
            input = input.clone().detach()
        else:
            input = torch.FloatTensor(input)
            
        if isinstance(output, torch.Tensor):
            output = output.clone().detach()
        else:
            output = torch.FloatTensor(output)
            
        if isinstance(type, torch.Tensor):
            type = type.clone().detach()
        else:
            type = torch.LongTensor(type)
            
        sheet_idx = torch.LongTensor([sheet_idx])
            
        return input, output, type, sheet_idx
    
class NoiseDataFiltered(Dataset):
    def __init__(self):
        pass
    
class NoiseDataBin(Dataset):
    def __init__(self, dir='../data', filename='data_final.xlsx', use_type = None, transform = None, num_bins=51):
        self.dir = dir
        self.filename = filename
        self.use_type = use_type
        self.transform = transform
        self.num_bins = num_bins
        try:
            all_sheets = pd.read_excel(os.path.join(self.dir, self.filename), sheet_name=None)
        except Exception as e:
            raise e
        self.dataFrame = pd.DataFrame()
        self.sheet_names = []
        for sheet_name, df in all_sheets.items():
            self.sheet_names.append(sheet_name)
            self.dataFrame = pd.concat([self.dataFrame, df], ignore_index=True)
        
        self.le = []
        keys = self.dataFrame.keys()
        if self.use_type:
            le = LabelEncoder()
            self.dataFrame[keys[0]] = le.fit_transform(self.dataFrame[keys[0]])
            print(le.fit_transform(['AD02','BYD HTH','E0Y-3Z3M','GEM','H37','H97D','KKL','M1E','MAR2 2Z','MAR2 EVA2','MBQ','MEB','NU2','SA5H','SRH','T1X','T2X RHD','X03']))
            self.le = le.classes_

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        keys = self.dataFrame.keys()
        input = [self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx]]
        output = [self.dataFrame[keys[len(keys)-1]][idx]]
        if self.transform is not None:
            input = self.transform(input)
        # Bin values
        bins = np.array(range(20, 70, 1))   # num_bins = len(bins) + 1  51
        bin_output = torch.LongTensor(np.digitize(output, bins))

        bins1 = np.array(range(20, 70, 3)) # num_bins = len(bins) + 1  18
        bin_output0 = torch.LongTensor(np.digitize(output, bins1))

        bins1 = np.array(range(20, 70, 11)) # num_bins = len(bins) + 1  6
        bin_output1 = torch.LongTensor(np.digitize(output, bins1))

        bins1 = np.array(range(20, 70, 24)) # num_bins = len(bins) + 1  4
        bin_output2 = torch.LongTensor(np.digitize(output, bins1))

        input = torch.tensor(input).to(torch.float32)
        output = torch.tensor(output).to(torch.float32)
        if self.use_type:
            type_ = torch.LongTensor(np.array(range(self.dataFrame[keys[0]][idx]*self.num_bins, (self.dataFrame[keys[0]][idx]+1)*self.num_bins)))
            return input, output, bin_output, bin_output0, bin_output1, bin_output2, type_
        else:
            return input, output, bin_output, bin_output0, bin_output1, bin_output2

class NoiseDataFFT(Dataset):
    def __init__(self, dir='../data', filename='data_final_fft_0217.xlsx', use_type = None, transform = None, debug = None, fft_out=401):
        self.dir = dir
        self.filename = filename
        self.use_type = use_type
        self.transform = transform
        self.debug = debug
        self.fft_out = fft_out
        try:
            if 'train' in filename:
                all_sheets = pd.read_excel(os.path.join(self.dir, 'data_final_fft_train_0217.xlsx'), sheet_name=None)
            elif 'test' in filename:
                all_sheets = pd.read_excel(os.path.join(self.dir, 'data_final_fft_test_0217.xlsx'), sheet_name=None)
            else:
                all_sheets = pd.read_excel(os.path.join(self.dir, self.filename), sheet_name=None)
        except Exception as e:
            raise e
        self.dataFrame = pd.DataFrame()
        self.sheet_names = []
        for sheet_idx, (sheet_name, df) in enumerate(all_sheets.items()):
            self.sheet_names.append(sheet_name)
            df['sheet_idx'] = sheet_idx
            self.dataFrame = pd.concat([self.dataFrame, df], ignore_index=True)
        
        self.le = []
        keys = self.dataFrame.keys()
        if self.use_type:
            le = LabelEncoder()
            self.dataFrame[keys[0]] = le.fit_transform(self.dataFrame[keys[0]])
            # print(le.fit_transform(['AD02','BYD HTH','E0Y-3Z3M','GEM','H37','H97D','KKL','M1E','MAR2 2Z','MAR2 EVA2','MBQ','MEB','NU2','SA5H','SRH','T1X','T2X RHD','X03']))
            self.le = le.classes_

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        keys = self.dataFrame.keys()
        input = [self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx]]
        M1 = [self.dataFrame[keys[4]][idx]]
        # output = self.dataFrame.iloc[idx, 5:].tolist()   # 0~25600 401维
        output = self.dataFrame.iloc[idx, 5:30].tolist()   # 31.5~10000 26维
        sheet_idx = self.dataFrame['sheet_idx'][idx]
        
        if self.transform is not None:
            input = self.transform(input)
        
        if isinstance(input, torch.Tensor):
            input = input.clone().detach()
        else:
            input = torch.FloatTensor(input)
        
        if isinstance(output, torch.Tensor):
            output = output.clone().detach()
        else:
            output = torch.FloatTensor(output)
        
        if isinstance(M1, torch.Tensor):
            M1 = M1.clone().detach()
        else:
            M1 = torch.FloatTensor(M1)
        
        # output = 10*(torch.log10(output/4e-10))
        
        if self.debug:
            return input, output, self.dataFrame.iloc[idx, 5:], M1
        if self.use_type:
            type_ = torch.LongTensor([self.dataFrame[keys[0]][idx]])
            return input, output, type_, sheet_idx
        else:
            return input, output, sheet_idx