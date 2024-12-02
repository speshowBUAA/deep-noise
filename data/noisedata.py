import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np

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
            self.le = le.classes_

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        keys = self.dataFrame.keys()
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

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass