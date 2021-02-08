import torch
from torch import nn
import numpy as np
from tqdm.notebook import tqdm

trans = lambda x: x.transpose(-1,-2)

class reservoir_net(nn.Module):
    def __init__(self,batches=1,input_size=1,rank=1,output_size=1,N=256,dt=0.1,noise_sig=0,phi_type="tanh",g=0.9,device='cpu',**kwargs):
        super().__init__()
        local_vars=locals().copy()
        [setattr(self,item,local_vars[item]) for item in local_vars.keys() if not item=='self']
        
        self.J    = torch.normal(0,g/np.sqrt(N),(batches,N,N),device=device)
        self.wIn  = torch.normal(0,1,(batches,N,input_size),device=device)
        self.wOut = torch.zeros((batches,N,output_size),device=device)
        self.m    = torch.normal(0,1,(batches,N,rank),device=device)
        self.n    = torch.zeros((batches,N,rank),device=device)
        self.P    = torch.eye(N,device=device)[None].repeat(batches,1,1)/10
        self.set_mnT()
        self.dt = dt
        self.phi_type = phi_type
        self.init_phi()
        self.store_errors = []
        self.bs = 5

    def set_mnT(self):
        self.mnT = self.m @ trans(self.n)
        
    def init_phi(self):
        phi_type=self.phi_type
        if phi_type=="linear":
            self.phi=nn.Identity()
        elif phi_type=='tanh':
            self.phi=nn.Tanh()
        elif phi_type=='relu':
            self.phi=nn.Relu()

    def recurrence(self,input,z=None):
        dt=self.dt
        noise = torch.randn(x.shape,device=input.device) * self.noise_sig  if self.noise_sig>0  else 0
        self.r = self.phi(self.x)
        self.z = trans(self.n) @ self.r if z is None else z
        self.x = (1-dt)*self.x + dt*(self.J @ self.r + self.m @ self.z + self.wIn @ input + noise)

    
    def init_data(self, input_shape):
        batch_size = input_shape[-1]
        self.x = torch.zeros((self.batches,self.N,batch_size),device=self.device)
        #return x

    def run(self,input,x=None,f=None,OL=False,use_tqdm=True,RLS=True,learn_every=1):
        if x is None:
            self.init_data(input.shape)
        all_x=[]
        all_z=[]
        steps = np.arange(input.shape[1])
        steps = tqdm(steps) if use_tqdm else steps
        for i in steps:
            self.recurrence(input[:,i], z = f[:,i,:] if OL else None)
            if RLS:
                self.RLS(f[:,i,:self.rank])
            x = self.x.detach().clone().cpu()
            z = self.z.detach().clone().cpu()
            all_x.append(x)
            all_z.append(z)
        all_x = torch.stack(all_x, dim=1)
        all_z = torch.stack(all_z, dim=1)
        return all_x,all_z
    
    def RLS(self,f):
        x,r,z = self.x,self.r,self.z
        P = self.P
        Pr = P @ r
        rPr = trans(r) @ Pr
        c = (1.0/(1.0 + rPr))
        P = P - c.repeat(1,self.N,self.N) * (Pr @  trans(Pr))
        self.P = P
        z = trans(self.n) @ r
        err = z - f
        dw = - c.repeat(1,self.N,err.shape[-2]) * (Pr @ trans(err))
        self.n = self.n + dw
        self.store_errors.append(err.detach().cpu())