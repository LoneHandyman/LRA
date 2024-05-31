import torch
import torch.nn as nn
import math
from torch.utils.data.dataset import Dataset
import sys
import os
import random
import json
import pickle
import numpy as np
from googleapiclient.discovery import build
import requests

api_key = 'AIzaSyBUJfIERM34j-wkgCz9DXix_5Bd6nab3h8'
folder_url = 'https://drive.google.com/drive/folders/1rorPn7uKUYFw4VPcTT-_tdZX-hUvZS28'

def download_file_from_google_drive(api_key, folder_url, file_path):
    # Extract the folder ID from the URL
    file_name = file_path.split('/')[-1]
    folder_id = folder_url.split('/')[-1]

    # Build the service
    service = build('drive', 'v3', developerKey=api_key)

    # Search for the file by name in the specified folder
    query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
        return

    # Assuming the first match is the desired file
    file_id = items[0]['id']
    download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"

    # Download the file content
    response = requests.get(download_url)

    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f'File {file_path} downloaded successfully.')
    else:
        print(f'Failed to download file. Status code: {response.status_code}')


class LRADataset(Dataset):
    def __init__(self, file_path, endless):

        self.endless = endless

        if not os.path.isfile(file_path):
            print(f'File {file_path} not found. Trying to download it.')
            download_file_from_google_drive(api_key, folder_url, file_path)
        
        with open(file_path, "rb") as f:
            self.examples = pickle.load(f)
            random.shuffle(self.examples)
            self.curr_idx = 0
            
        print(f"Loaded {file_path}... size={len(self.examples)}", flush = True)

    def __len__(self):
        if self.endless:
            return 1000000000
        else:
            return len(self.examples)

    def create_inst(self, inst):
        output = {}
        output["input_ids_0"] = torch.tensor(inst["input_ids_0"], dtype = torch.long)
        output["mask_0"] = (output["input_ids_0"] != 0).float()
        if "input_ids_1" in inst:
            output["input_ids_1"] = torch.tensor(inst["input_ids_1"], dtype = torch.long)
            output["mask_1"] = (output["input_ids_1"] != 0).float()
        output["label"] = torch.tensor(inst["label"], dtype = torch.long)
        return output
    
    def __getitem__(self, i):
        if not self.endless:
            return self.create_inst(self.examples[i])
        
        if self.curr_idx >= len(self.examples):
            random.shuffle(self.examples)
            self.curr_idx = 0
        inst = self.examples[self.curr_idx]
        self.curr_idx += 1
        
        return self.create_inst(inst)
