import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from typing import Tuple
import pickle
from os import path
import glob

import random

from data_transforms import (
    MinMaxTransform1D,
    LogTransform1D,
    SymLogTransform1D,
    DataTransform
)

# The code below is adapted from Cosmin's bcs_evm.ipynb
#
#

# dataset constants 
DATA_PATH = '/projects/wg-moltensalt/HT9'
INPUT_NAMES = ['vmJ2', 'temperature', 'evm', 'rhom', 'rhoi', 'flux']
OUTPUT_NAMES = ['evm', 'rhom', 'rhoi']

def load_filtered_simulations(mode: str, samples_per_sim = 500):
    '''
    Extract simulations from the HT9 dataset:
    Inputs:
        mode = 'train', 'valid, or 'test'
        
    Returns:
        dict of simulations
    '''
        
    fpattern = f'{DATA_PATH}/{mode}*.pickle'
    files = [path.basename(x) for x in glob.glob(fpattern)]
    
    simdata = {'mode': mode, 'input_vars': INPUT_NAMES, 'output_vars': OUTPUT_NAMES, 'sim': {}}
    count = 0
    
    for file_name in files:
        split_name = file_name.split('_')
        file_mode = split_name[0]
        # if file_mode != mode[0]:
        #     continue
        file_ext = file_name.split('.')[-1]
        file_type = split_name[-1].split('.')[0][-1]
        # file_type = 'd' is metadata, 'a' is mechanism contribution
        
        if file_type != 'a' and file_type != 'd' and file_type != 'r' and file_ext != 'dvc':
            #load data
            with open(f'{DATA_PATH}/{file_name}', 'rb') as f:
                #print(DATA_PATH+'/'+file_name)
                d = pickle.load(f)
                
            # load corresponding meta data
            meta_name = file_name.split('_')
            num_sims, ext = meta_name[-1].split('.')
            meta_name[-1] = num_sims+'d.'+ext
            meta_name = '_'.join(meta_name)    
            with open(f'{DATA_PATH}/{meta_name}', 'rb') as f:
                #print(data_path+'/'+meta_name)
                metadata = pickle.load(f)
                 
            for k in range(len(d[0])): # looping over number SIMULATION INDICES  
                # added test below due to "train_HT9_j2sc12c_493.pickle" which had a time slot empty
                if len(d[0][k])==0 or len(d[1][k])==0:
                    continue
                
                X = np.concatenate([d[1][k][q][:,None] for q in INPUT_NAMES],axis=1) #ITERATION TRACES of VARIABLE q
                Y = np.concatenate([d[0][k][q][:,None] for q in OUTPUT_NAMES],axis=1)
                dt = np.array(metadata[0][k]['inc'])
                xinit = np.array([metadata[0][k][name] for name in INPUT_NAMES])
                assert(dt.shape[0] == X.shape[0])

                
                # remove nans
                if np.isnan(X).any() or np.isnan(Y).any() or np.isnan(dt).any():
                    idx = np.unique(np.argwhere(np.isnan(X))[:,0])
                    X = np.delete(X, idx, axis=0)
                    Y = np.delete(Y, idx, axis=0)
                    dt = np.delete(dt, idx, axis=0)

                    idx = np.unique(np.argwhere(np.isnan(Y))[:,0])
                    X = np.delete(X, idx, axis=0)
                    Y = np.delete(Y, idx, axis=0)
                    dt = np.delete(dt, idx, axis=0)

                    idx = np.unique(np.argwhere(np.isnan(dt))[:,0])
                    X = np.delete(X, idx, axis=0)
                    Y = np.delete(Y, idx, axis=0)
                    dt = np.delete(dt, idx, axis=0)

                # remove values where evm_dot <= 0
                if np.any(Y[:,0]<1.e-100):
                    idx = np.where(Y[:,0]<1.e-100)[0]
                    X = np.delete(X, idx, axis=0)
                    Y = np.delete(Y, idx, axis=0)
                    dt = np.delete(dt, idx, axis=0)

                # subsample uniformly in log10(t)
                
                t = np.cumsum(dt)
                nt = t.shape[0]
                ns = min(samples_per_sim,nt)
                logt = np.log10(t) 
                maxt = logt.max()
                mint = logt.min()
                tbnds = np.linspace(mint, maxt, ns, endpoint=False)

                sidx = []
                s = 0
                for i in range(nt):
                    if logt[i] >= tbnds[s]:
                        sidx.append(i)
                        s += 1
                    if s == ns:
                        break

                X = X[sidx,:]
                Y = Y[sidx,:]
                dt = dt[sidx]
                
                ## filtering over.
                simdata['sim'][count] = {'xinit': xinit, 'x': X, 'y': Y, 'dt': dt}
                count+=1
                
    return simdata

def load_simulations(mode: str):
    '''
    Extract simulations from the HT9 dataset:
    Inputs:
        mode = 'train', 'valid, or 'test'
        
    Returns:
        dict of simulations
    '''
        
    fpattern = f'{DATA_PATH}/{mode}*.pickle'
    files = [path.basename(x) for x in glob.glob(fpattern)]
    
    simdata = {'mode': mode, 'input_vars': INPUT_NAMES, 'output_vars': OUTPUT_NAMES, 'sim': {}}
    count = 0
    
    for file_name in files:
        split_name = file_name.split('_')
        file_mode = split_name[0]
        # if file_mode != mode[0]:
        #     continue
        file_ext = file_name.split('.')[-1]
        file_type = split_name[-1].split('.')[0][-1]
        # file_type = 'd' is metadata, 'a' is mechanism contribution
        
        if file_type != 'a' and file_type != 'd' and file_type != 'r' and file_ext != 'dvc':
            #load data
            with open(f'{DATA_PATH}/{file_name}', 'rb') as f:
                #print(DATA_PATH+'/'+file_name)
                d = pickle.load(f)
                
            # load corresponding meta data
            meta_name = file_name.split('_')
            num_sims, ext = meta_name[-1].split('.')
            meta_name[-1] = num_sims+'d.'+ext
            meta_name = '_'.join(meta_name)    
            with open(f'{DATA_PATH}/{meta_name}', 'rb') as f:
                #print(DATA_PATH+'/'+meta_name)
                metadata = pickle.load(f)
                 
            for k in range(len(d[0])): # looping over number SIMULATION INDICES  
                # added test below due to "train_HT9_j2sc12c_493.pickle" which had a time slot empty
                if len(d[0][k])==0 or len(d[1][k])==0:
                    continue
                
                X = np.concatenate([d[1][k][q][:,None] for q in INPUT_NAMES],axis=1) #ITERATION TRACES of VARIABLE q
                Y = np.concatenate([d[0][k][q][:,None] for q in OUTPUT_NAMES],axis=1)
                dt = np.array(metadata[0][k]['inc'])
                xinit = np.array([metadata[0][k][name] for name in INPUT_NAMES])
                
                assert(dt.shape[0] == X.shape[0])

                #### preprocess simulations
                ## remove nans
                if np.isnan(X).any() or np.isnan(Y).any() or np.isnan(dt).any():
                    idx = np.unique(np.argwhere(np.isnan(X))[:,0])
                    X = np.delete(X, idx, axis=0)
                    Y = np.delete(Y, idx, axis=0)
                    dt = np.delete(dt, idx, axis=0)

                    idx = np.unique(np.argwhere(np.isnan(Y))[:,0])
                    X = np.delete(X, idx, axis=0)
                    Y = np.delete(Y, idx, axis=0)
                    dt = np.delete(dt, idx, axis=0)

                    idx = np.unique(np.argwhere(np.isnan(dt))[:,0])
                    X = np.delete(X, idx, axis=0)
                    Y = np.delete(Y, idx, axis=0)
                    dt = np.delete(dt, idx, axis=0)

                # remove values where evm_dot ~ 0 or < 0
                if np.any(Y[:,0]<1.e-100):
                    idx = np.where(Y[:,0]<1.e-100)[0]
                    X = np.delete(X, idx, axis=0)
                    Y = np.delete(Y, idx, axis=0)
                    dt = np.delete(dt, idx, axis=0)

                #simdata['sim'] = {}
                simdata['sim'][count] = {'xinit': xinit, 'x': X, 'y': Y, 'dt': dt}
                count+=1
                
    return simdata

def load_data(mode: str):
    '''
    Load HT9Dataset.
    Inputs:
        mode = 'train' or 'test'
        
    Returns:
        x,y
    '''
    simdata = load_simulations(mode=mode)
    X = np.concatenate([simdata['sim'][i]['x'] for i in simdata['sim'].keys()], axis=0)
    Y = np.concatenate([simdata['sim'][i]['y'] for i in simdata['sim'].keys()], axis=0)                   
    return X, Y

def load_filtered_data(mode: str, start_at=0):
    '''
    Load HT9Dataset.
    Inputs:
        mode = 'train' or 'test'
        
    Returns:
        x,y
    '''
    simdata = load_filtered_simulations(mode=mode, samples_per_sim=500)
    X = np.concatenate([simdata['sim'][i]['x'][start_at:] for i in simdata['sim'].keys()], axis=0)
    Y = np.concatenate([simdata['sim'][i]['y'][start_at:]  for i in simdata['sim'].keys()], axis=0)                   
    return X, Y

def load_data_start_at(mode: str, start_at=10):
    '''
    Load HT9Dataset, but ignore the first "start_at" iterations.
    
    Inputs:
        mode = 'train' or 'test'
        
    Returns:
        x,y
    '''
    simdata = load_simulations(mode=mode)
    X = np.concatenate([simdata['sim'][i]['x'][start_at:,:] for i in simdata['sim'].keys()], axis=0)
    Y = np.concatenate([simdata['sim'][i]['y'][start_at:,:] for i in simdata['sim'].keys()], axis=0)                   
    return X, Y

def load_simple(mode: str, start_at=10):
    simdata = load_simulations(mode=mode)
    X = np.concatenate([simdata['sim'][i]['x'][start_at:,:] for i in simdata['sim'].keys()], axis=0)
    Y = np.concatenate([simdata['sim'][i]['y'][start_at:,:] for i in simdata['sim'].keys()], axis=0)  
    
    idx = np.unique(np.argwhere( X[:,3] < 1e11 )[:,0])
    X = np.delete(X, idx, axis=0)
    Y = np.delete(Y, idx, axis=0)

    idx = np.unique(np.argwhere( Y[:,0] < 1e-10 )[:,0])
    X = np.delete(X, idx, axis=0)
    Y = np.delete(Y, idx, axis=0)
    
    return X, Y

def generate_data_transform(x: np.array, y: np.array):
    ## Init transforms
    input_transforms_list = [
        MinMaxTransform1D(),
        MinMaxTransform1D(),
        LogTransform1D(lb=-1.0,ub=1.0, eps=1e-20),
        LogTransform1D(lb=-1.0,ub=1.0), 
        LogTransform1D(lb=-1.0,ub=1.0),
        MinMaxTransform1D(),
    ]

    output_transforms_list = [
        LogTransform1D(lb=0,ub=1.0), 
        SymLogTransform1D(), 
        SymLogTransform1D(), 
    ]

    input_transform = DataTransform(input_transforms_list)
    output_transform = DataTransform(output_transforms_list)
    
    input_transform.fit(x)
    output_transform.fit(y)

    data_transform = {
                    'input': input_transform,
                    'output': output_transform,
                    }
    return data_transform

def filtered_train_test_dataset():
    '''
    Constructs two dataset (head, tail) based on threshold t, applied to the output evm_dot
    '''
    xtr, ytr = load_filtered_data('train') 
    xte, yte = load_filtered_data('test')

    data_transform = generate_data_transform(xtr, ytr)

    #form datasets
    opts_tr = {'mode': 'train', 'data_transform': data_transform}
    opts_te = {'mode': 'test', 'data_transform': data_transform}
            
    ds_train = HT9Dataset(xtr, ytr, opts_tr)
    ds_test = HT9Dataset(xte, yte, opts_te)
    return ds_train, ds_test

def train_test_dataset():
    '''
    Constructs two dataset (head, tail) based on threshold t, applied to the output evm_dot
    '''
    xtr, ytr = load_data('train') 
    xte, yte = load_data('test')

    data_transform = generate_data_transform(xtr, ytr)

    #form datasets
    opts_tr = {'mode': 'train', 'data_transform': data_transform}
    opts_te = {'mode': 'test', 'data_transform': data_transform}
            
    ds_train = HT9Dataset(xtr, ytr, opts_tr)
    ds_test = HT9Dataset(xte, yte, opts_te)
    return ds_train, ds_test

def multi_loader(batch_size: int, num_workers = 1):
    '''
    Splits HT9 dataset and constructs multiple loaders
    ''' 
    
    # xtr, ytr = load_data('train')
    # xte, yte = load_data('test')

    xtr, ytr = load_simple('train')
    xte, yte = load_simple('test')
    
    bounds = [-np.inf,1e-5, 1e-1,np.inf] #currently manually tuned and applied only to evm_dot
    nb = len(bounds)-1
    probs = [0.4, 0.35, 0.25]
    
    batch_sizes = [int(p*batch_size) for p in probs[:-1]] 
    batch_sizes += [batch_size - sum(batch_sizes)]
    
    data_transform = generate_data_transform(xtr, ytr)

    #form datasets
    opts_tr = {'mode': 'train', 'data_transform': data_transform}
    opts_te = {'mode': 'test', 'data_transform': data_transform}
    
    train_loaders = []
    test_loaders = []
    
    for ii in range(nb):
        idx = np.asarray((bounds[ii] <= ytr[:,0]) & (ytr[:,0] < bounds[ii+1]) ).nonzero()[0]
        assert idx.size != 0, f"empty dataset"
        train_loader = DataLoader(
            HT9Dataset(xtr[idx,:], ytr[idx,:], opts_tr),
            batch_size=batch_sizes[ii],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        train_loaders.append( train_loader )
        
        idx = np.asarray((bounds[ii] <= yte[:,0]) & (yte[:,0] < bounds[ii+1]) ).nonzero()[0]
        assert idx.size != 0, f"empty dataset"
        test_loader = DataLoader(
            HT9Dataset(xte[idx,:], yte[idx,:], opts_te),
            batch_size=batch_sizes[ii],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loaders.append( test_loader )
    

    return train_loaders, test_loaders



class HT9Dataset(Dataset):

    def __init__(self, x: torch.tensor, y: torch.tensor, opts: dict):
        '''
        Dataset object for the HT9 dataset, 
        '''
        self.data_path = DATA_PATH
        self.data_transform = opts['data_transform']
        self.mode = opts['mode']
        self.input_names = INPUT_NAMES
        self.output_names = OUTPUT_NAMES
                   
        self.x = self.data_transform['input'].transform(x)
        self.y = self.data_transform['output'].transform(y)
    

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index: int):
        '''
        Returns data points
        '''
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])
    
    def transform_input(self, x: np.array):
        return self.data_transform['input'].transform(x)

    def inverse_transform_input(self, x: np.array):
        return self.data_transform['input'].inverse_transform(x)
    
    def transform_output(self, y: np.array):
        return self.data_transform['output'].transform(y)

    def inverse_transform_output(self, y: np.array):
        return self.data_transform['output'].inverse_transform(y)




