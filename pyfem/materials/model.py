from collections import OrderedDict

import torch
import torch.nn as nn

## EXPERTS 
class MLP(nn.Module):
    def __init__(self, layer_dims: list):
        '''
        MLP with ReLU activations
        '''
        super(MLP, self).__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(self.layer_dims)-2):
            layers.extend([nn.Linear(in_features=self.layer_dims[i], out_features=self.layer_dims[i+1], bias=True),
                           nn.LayerNorm((self.layer_dims[i+1],)),
                           nn.SiLU(),
                           nn.Dropout(p=0.5),
                          ])
        layers.extend([nn.Linear(in_features=self.layer_dims[-2], out_features=self.layer_dims[-1], bias=True)])
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.sequential(x)


class LinearHN(nn.Module):
    def __init__(self, in_features: int, features: int, out_features: int):
        '''
        Linear Hypernetwork expert -- Linear experts, where the weights are generated by a MLP: 
        pred = x @ W(x) + b(x) + x @ A + b
        '''
        super(LinearHN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.features = features
        self.hn = MLP( layer_dims = [self.in_features, self.features, self.features, self.features, self.in_features * self.out_features + self.out_features] )
        self.linear = nn.Linear(self.in_features, self.out_features, bias=True)
        
    def forward(self, x: torch.tensor):
        # x is (B,in_features)
        out = self.hn( x ) #(B, in_features * out_features + out_features)
        bx = out[:,-self.out_features:] #(B, out_features)
        Wx = out[:,:-self.out_features].view(-1, self.in_features, self.out_features) #(B, in_features, out_features)
        return torch.bmm(x[:,None,:], Wx).squeeze(dim=1) + bx + self.linear(x)

class SoftmaxGate(nn.Module): #To do: expand to MLP + Softmax Gate
    '''
    SOFTMAX Gate
    '''
    def __init__(self, in_features: int, gate_features: int, num_experts: int, temp: float = 1.0):
        super(SoftmaxGate, self).__init__()
        self.in_features = in_features
        self.gate_features = gate_features
        self.num_experts = num_experts
        self.linear_gate = LinearHN(self.in_features, self.gate_features, self.num_experts)
        
        self.softmax = nn.Softmax(dim=-1)
        self.temp = temp

        self.load = None 

    def forward(self, x: torch.tensor):
        # x is (B, d)
        
        z = self.linear_gate( x ) 
        probs = self.softmax( z / self.temp ) #(B, num_experts)

        
        expert_probs = probs.mean(dim=0) #probility that an expert is selected, across the batch --> "expert usage"
        # self.load = self.num_experts*(expert_probs**2).sum()
        # WHY this definition of load?
        # suppose p is a (n,) probability distribution. Then p.sum() == 1 and p lies in the simplex.
        # the smallest ball in R^n that touches the simplex touches at q_i = 1/n for i=1,...,n. Hence, 
        # (p**2).sum() >= (q**2).sum() = 1/n**2 + ... + 1/n**2 = n * 1/n**2 = 1/n
        #
        # We want to include a load balancing loss to handle outliers, i.e., to ensure that outliers are allocated some portion of the experts.
        # We must be cautious that this does not harm expert specialization / sparsity of the softmax distribution though...
        self.load = self.num_experts*(expert_probs**2).sum() + (1-(probs**2).sum(dim=1)).mean()
        
        return probs
        
class MixtureOfExperts(nn.Module):
    '''
    Mixture of Experts in a lifted space.
    Preprocessing step first lifts, reorients, and then normalizes the input data.
    Gating and Experts constructed in this lifted space.
    '''
    
    def __init__(self, in_features: int, gate_features: int, expert_features: int, out_features: int, num_experts: int, device='cpu'):
        super(MixtureOfExperts, self).__init__()
        self.in_features = in_features
        self.gate_features = gate_features
        self.expert_features = expert_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.device = device

        self.gate = SoftmaxGate(self.in_features, self.gate_features, self.num_experts).to(self.device)
        self.experts = nn.ModuleList([LinearHN(self.in_features, self.expert_features, self.out_features).to(self.device) for i in range(self.num_experts)])
        
    def forward(self, x: torch.tensor):
        # x is (B, n)
        g = self.gate(x)[:,:,None] #(B, num_experts,1)
        f = torch.cat([m(x)[:,None,:] for i,m in enumerate(self.experts)], dim=1) #(B, num_experts, 3)

        return (g*f).sum(dim=1)