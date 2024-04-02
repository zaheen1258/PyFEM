#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import sys
import copy
import os
import matplotlib.pyplot as plt
import glob
import torch
from pyfem.materials import model


# In[2]:
sys.path.insert(0, '/Users/zaheennasir/mypyfem/PyFEM/pyfem/materials')

## loading pytorch model

model_name = f"/Users/zaheennasir/mypyfem/PyFEM/pyfem/materials/model_experiment_5_1101443/checkpoint_savetime_1101443_batchsize_16384_numexperts_64.pt"
device = 'cpu'
num_experts = int( model_name.split('_')[-1][:-3] )
num_vars = 6
gate_features = 128
expert_features = 32
out_features = 3
model1 = model.MixtureOfExperts(num_vars, gate_features, expert_features, out_features, num_experts)
model1.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
model1.to(torch.double)
model1.to(device)
model1.eval()


# In[3]:


with open(f"/Users/zaheennasir/mypyfem/PyFEM/pyfem/materials/transform.pickle","rb") as fhandle:
    data_transform = pickle.load(fhandle)

with open(f"/Users/zaheennasir/mypyfem/PyFEM/pyfem/materials/init_data.pickle","rb") as fhandle:
    init_data = pickle.load(fhandle)


# In[4]:


## surrogate eval and jacobian wrapper, for pointwise evaluations

class ModelWrapper:

    def __init__(self, model, data_transform, init_data):
        self.data_transform = data_transform
        self.model = model
        self.init_data = init_data
        
        self.ninputs = self.data_transform['input'].length
        self.noutputs = self.data_transform['output'].length

    def eval(self, x: np.array):
        # x is (n,)
        x_extended = np.array([x])
        z = torch.tensor(self.data_transform['input'].transform(x_extended), dtype=torch.double)         
        # eval surrogates              
        with torch.no_grad():
            outs  = self.model(z)
        return self.data_transform['output'].inverse_transform(outs.numpy()).reshape((-1,))

    def eval_jacobian(self, x: np.array):
        '''
        pytorch jacobian modified from
        https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
        '''
        # x is (n,)

        x_extended = np.array([x])
        Jin = self.data_transform['input'].forward_derivative(x_extended)
        z = torch.tensor( self.data_transform['input'].transform(x_extended), dtype=torch.double).squeeze()
        z = z.repeat(self.noutputs, 1)
        z.requires_grad_(True)
        y = self.model(z)
        y.backward(torch.eye(self.noutputs))
        Jmod = z.grad.data.numpy()
        Jout = data_transform['output'].inverse_derivative( y[0].detach().numpy()[None,:] )
        J = Jout[:,None]*(Jmod * Jin) # (3,6) jacobian for all 6 inputs
        return J # (3,3) jacobian for just the state variables 

    def eval_func_wrapper(self, t, x):
        '''
        wrapper for scipy.integrate.solve_ivp
        '''
        return self.eval(x)

    def eval_jacobian_wrapper(self, t, x):
        '''
        wrapper for scipy.integrate.solve_ivp
        '''
        return self.eval_jacobian(x)
    


# In[5]:


Phi = ModelWrapper(model1, data_transform, init_data)


# In[7]:


'''set1 = np.array([0.119132960099, 600.0690046472204, 0.0, 4669.9511535360325, 4406641771830.477, 1.018454806316354e-09])
Phi.eval(set1)'''


# In[ ]:




