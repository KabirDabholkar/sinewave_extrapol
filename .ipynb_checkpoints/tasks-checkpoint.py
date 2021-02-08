import numpy as np
from itertools import product
"""
List of functions for several tasks. Some are autononous - so the functions only return only targ.

Now use sine_wave_task class for gaussian and phase split versions.
"""

class sine_wave_task:
    """
    A class to generate training and testing data for
    the sine wave task. 
    """
    def __init__(self,
                 split=1,
                 dt=0.1,
                 min_bout=int(15), #seconds
                 tau_bout=int(15), #seconds
                 fix=20,
                 inp_range=np.arange(0.05,0.1,0.01),
                 freq_func=lambda omega:omega,
                 inp_split_style="original",
                 out_split_style="phase",
                 rand_init_phase=False,
                 sigma=2):

        local_vars=locals().copy()
        [setattr(self,item,local_vars[item]) for item in local_vars.keys() if not item=='self']

        """
        inp_split_func takes input value as input and returns
        multichannel input.
        """
        if inp_split_style=="original":
            self.inp_dim=1
            self.inp_split_func=lambda x: x
            self.inp_to_val=lambda llinp: inp[0,:]
        elif inp_split_style=="multichannel_switch":
            self.inp_dim=len(self.inp_range)
            self.inp_split_func=lambda x: (self.inp_range==x).reshape(len(self.inp_range),1).astype(np.float32)
            self.inp_to_val=lambda inp: np.dot(inp_range,inp)
        else:
            raise Exception("inp_split_style must be one of 'original' or 'multichannel_switch'.")        

    def out_split_func(self,seq,split=None,style=None,phases=None):
        """
        out_split_func takes sequence of phases as input and
        outputs the corresponding split target sequence.
        """
        out_split_style = self.out_split_style if style==None else style
        split = self.split if split==None else split
        if out_split_style=="phase":
            phases=phases if phases is not None else np.linspace(0,np.pi,split+1)[:split]
            return np.cos(np.tile(phases,(len(seq),1)).T+np.tile(seq,(split,1)))
        elif out_split_style=="gaussian_sine":
            field_centers=np.linspace(-1,1,split)
            sigma=sigma
            return np.exp(-(np.tile(np.sin(seq),(split,1))\
                                -np.tile(field_centers,(len(seq),1)).T)**2/sigma**2)
        elif out_split_style=='gaussian':
            field_centers=np.linspace(0,1,self.split+1)[:self.split]
            sigma=sigma
            distance_to_nearest_int = lambda x:np.min(np.stack([x-np.floor(x),np.ceil(x)-x]),axis=0)
            difference = lambda seq: np.tile(np.mod(seq/(2*np.pi),1),(split,1)) \
                                -np.tile(field_centers,(len(seq),1)).T
            return np.exp(-distance_to_nearest_int(difference(seq))**2/sigma**2)
        else:
            raise Exception("'style' must be one of 'gaussian or 'phase' or 'gaussian_sine.") 
            
    def generate(self,steps,input_val=None,split=None,style=None,phases=None):
        split = self.split if split==None else split
        inpt = np.zeros((self.inp_dim,steps))
        targ = np.zeros((split,steps))
        phase = np.random.rand(1)*np.pi*2 if self.rand_init_phase else 0
        t=self.fix
        
        while t<steps:
            duration=int(np.ceil(np.random.exponential(int(self.tau_bout/self.dt))+int(self.min_bout/self.dt))) if input_val==None else steps
            rval = 0 if t==0 else np.random.randint(0,len(self.inp_range))
            value=self.inp_range[rval] if input_val==None else input_val
            split_inp=self.inp_split_func(value)
            stop=min(steps,t+duration)
            inpt[:,t:t+duration]=split_inp
            seq=np.arange(stop-t)*2*np.pi*self.freq_func(value)*self.dt+phase
            seq_all=self.out_split_func(seq,split=split,style=style,phases=phases)
            targ[:,t:stop]=seq_all
            phase=seq[-1]
            t+=duration
        return inpt,targ
    
    def generate_many(self,steps,many,input_val=None,split=None,style=None,phases=None):
        split = self.split if split==None else split
        inpt=np.zeros((many,steps,self.inp_dim))
        targ=np.zeros((many,steps,split))        
        
        for i in range(many):
            val=input_val[i] if type(input_val) is np.ndarray else None
            inp,tar=self.generate(steps,input_val=val,split=split,style=style,phases=phases)
            inpt[i,:,:],targ[i,:,:]=inp.T,tar.T
        #inpt,targ=map(lambda x: torch.tensor(x.astype(np.float32)),[inpt,targ])
        return inpt,targ
    
    def cycle_duration(self,cycles,input_val):
        """
        Returns the number of steps to complete
        the given number of cycles at the given
        input_val.
        """
        time_steps=np.round(cycles/self.freq_func(input_val)/self.dt).astype(np.int)
        return time_steps

    def set_dt_by_freq(self,freq,dt_T_ratio=0.01):
        T=1/freq
        self.dt=dt_T_ratio*T