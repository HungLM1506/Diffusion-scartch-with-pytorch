#Import lib
import torch 
import torch.nn as nn 
import torchvision
from tqdm import tqdm




#Class Diffusion
class Diffusion:
    def __init__(self,noise_steps=1000,beta_start=1e-4,beta_end=0.02,device='cpu',img_size=256):
        self.noise_steps=1000
        self.beta_start=beta_start
        self.beta_end=beta_end
        self.device=device
        self.img_size=img_size

        self.beta=self.prepare_beta_schedule().to(self.device)
        self.alpha=1-self.beta
        self.alpha_hat=torch.cumprod(self.alpha,dim=0) #calculate cumprod

    def prepare_beta_schedule(self):
        return torch.linspace(self.beta_start,self.beta_end,self.noise_steps)

    def noise_image(self,x,t):
         """
         x is input image
         t is timestep
         """
         sqrt_alpha_hat=torch.sqrt(self.alpha_hat[t])[:,None,None,None]
         sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:,None,None,None]
         e=torch.randn_like(x) #esilon
         return sqrt_alpha_hat*x+sqrt_one_minus_alpha_hat*e,e

    def sample_timesteps(self,n):
        return torch.randint(1,self.noise_steps,size=(n,))
    
    def sample(self,model,n):
        print(f'Sampling {n} new image...')
        model.eval()
        with torch.no_grad():
            x=torch.rand((n,3,self.img_size,self.img_size)).to(self.device)
            for i in tqdm(reversed(self.noise_steps,1)):
                t=(torch.ones(n)*i).long().to(self.device)
                predicted_noise=model(x,t)
                alpha=self.alpha[t]
                alpha_hat=self.alpha_hat[t]
                beta=self.beta[t]
                if t>1:
                    noise=torch.rand_like(x)
                else:
                    noise=torch.zeros_like(x)
                x=1/alpha_hat * (x-(1-alpha_hat)/torch.sqrt(1-alpha_hat) * predicted_noise) + torch.sqrt(beta)*noise
        return x






