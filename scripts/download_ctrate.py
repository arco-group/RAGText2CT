import shutil
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import argparse
import os 

def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files.
    """
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files

def download():

    split = 'train' # or valid
    batch_size = 100
    start_at = 0
    
    repo_id = 'ibrahimhamamci/CT-RATE'
    directory_name = f'dataset/{split}_fixed/'

    hf_token = '' # your hf_tokej
    
    data = pd.read_csv('train_labels.csv') 
        
    not_found = {"name":[]}
    
    for i in tqdm(range(start_at, len(data), batch_size), leave=False):
        data_batched = data[i:i+batch_size]
        for name in data_batched['VolumeName']:
            try:
                folder1 = name.split('_')[0]
                folder2 = name.split('_')[1]
                folder = folder1 + '_' + folder2
                folder3 = name.split('_')[2]
                subfolder = folder + '_' + folder3
                subfolder = directory_name + folder + '/' + subfolder
                if split == 'validation':
                    subfolder = subfolder.replace('validation', 'valid')
            
                hf_hub_download(repo_id=repo_id,
                    repo_type='dataset',
                    token=hf_token,
                    subfolder=subfolder,
                    filename=name,
                    cache_dir='./',
                    local_dir='dataset'
                    )
            except:
                not_found['name'].append(name)
                
    not_found = pd.DataFrame(not_found)
    not_found.to_csv(f'not_found.csv')
            
                
if __name__ == "__main__":
    download()