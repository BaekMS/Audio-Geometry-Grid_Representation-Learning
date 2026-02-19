# this is from torch-stft: https://github.com/pseeth/torch-stft
# this is from ConferencingSpeech2021: https://github.com/ConferencingSpeech/ConferencingSpeech2021

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch as th
import torch
import numpy as np
from scipy.signal import get_window

EPSILON = th.finfo(th.float32).eps
MATH_PI = math.pi

def init_kernels(win_len,
                 win_inc,
                 fft_len,
                 win_type=None,
                 invers=False):
    if win_type == 'None' or win_type is None:
        # N 
        window = np.ones(win_len)
    else:
        # N
        window = get_window(win_type, win_len, fftbins=True)#**0.5
   
    N = fft_len
    # N x F
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    # N x F
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    # 2F x N
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    # 2F x N * N => 2F x N
    kernel = kernel*window
    # 2F x 1 x N
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))

class LearnableNuDFT(nn.Module):
    def __init__(
        self,
        win_len: int,
        hop_len: int,      
        K: int,  # if None, use fft_len//2+1 
        vad_threshold=2/3,
        win_type: str = "hann",
        fs: float = 16000.0,             
        a_k_trainable: bool = False,
        eps_0=0.01,
        eps_1=100,
        nu_0: float = 0.0,        
        eps_start: float = 0.15,
        eps_end: float = 0.95,
    ):
        super().__init__()

        self.win_len = win_len
        self.stride = hop_len
        self.win_type = win_type
        self.fs = fs
        self.eps_0 = eps_0
        self.eps_1 = eps_1
        self.nu_0 = nu_0  # initial value for nu_0
        self.K = K  # number of frequency bins

        a_k_list= self.a_k_init(eps_start, eps_end)

        self.a_k_parameter = nn.Parameter(torch.tensor(a_k_list, dtype=torch.float32), requires_grad=a_k_trainable)  # (K-1,)  

        # window (for the time-domain window)
        if self.win_type is None or self.win_type.lower() == 'none':
            window_np = np.ones(self.win_len, dtype=np.float32)
        else:
            window_np = get_window(self.win_type, self.win_len, fftbins=True).astype(np.float32)
        self.register_buffer("window", torch.from_numpy(window_np))  # (win_len,)

        # precompute n for basis exponent: 0..N-1
        n = torch.arange(self.win_len, dtype=torch.float32)
        self.register_buffer("n", n)  # used for basis power

        self.vad_threshold=vad_threshold

        vad_kernel=torch.ones((1,1, self.win_len), dtype=torch.float32)/self.win_len
        self.register_buffer('vad_kernel', vad_kernel)

    def get_vad_framed(self, vad):
        N, P, L=vad.shape
        vad=vad.view(N*P, 1, L)
        pad_size=(self.stride-L)%self.stride
        vad=F.pad(vad, [0, pad_size])
        vad=F.conv1d(vad, self.vad_kernel, stride=self.stride)
        vad=vad.view(N, P, -1).ge(self.vad_threshold).long()
        return vad

    def a_k_init(self, eps_0, eps_1):
        
        numbering= np.linspace(eps_0, eps_1, num=self.K , dtype=np.float32, endpoint=True)

        logit_init = np.log(numbering / (1.0 - numbering))

        logit_init = logit_init - logit_init[0]
        logit_init =logit_init / logit_init[-1] * (self.win_len / 2.0)

        a_k_list = logit_init[1:] - logit_init[:-1]
        return a_k_list.tolist()
    
    def _compute_nu(self):
        """
        Compute non-uniform bin indices nu_k (in DFT bin units),
        cumulative sum of positive increments, normalized so nu_{K-1}=fft_len/2.
        """
        
        temp_a_k_data = self.a_k_parameter.data.clone()
        temp_a_k_data = torch.clamp(temp_a_k_data, min=self.eps_0, max=self.eps_1)  # ensure a_k_data is non-negative

        temp_a_k_sum = torch.sum(temp_a_k_data)

        if temp_a_k_sum > self.win_len / 2.0: # normalize to win_len/2  
            self.a_k_parameter.data = temp_a_k_data * (self.win_len / 2.0 / temp_a_k_sum)    

        a_k_cumsum = torch.cumsum(self.a_k_parameter, dim=0)  
        nu = a_k_cumsum + self.nu_0  # (K-1,)

        nu = torch.cat([torch.tensor([self.nu_0], device=nu.device), nu], dim=0)  # (K,)
        
        return nu  # (K,)
    
    def get_trajectory_framed(self, trajectory):
        B, spk_n, cart_dim, T=trajectory.shape # B, spk_n, 3, T
        trajectory=trajectory.contiguous()
        trajectory=trajectory.view(B*spk_n*cart_dim,1, T)
        pad_size=(self.stride-T)%self.stride
        trajectory=F.pad(trajectory, [0, pad_size], mode='replicate')

        trajectory=F.conv1d(trajectory, self.vad_kernel, stride=self.stride)

        trajectory=trajectory.view(B, spk_n, cart_dim, -1)

        return trajectory

    def _build_kernels(self):
        """
        Build complex basis kernels for NUDFT: exp(-j 2π * n * nu_k / N)
        Output: real+imag stacked kernels of shape (2K, 1, fft_len)
        """
        nu = self._compute_nu()  # (K,)   
        angle = -2 * math.pi * (nu[:, None] * self.n[None, :] / self.win_len)  # (K, fft_len)
        cos_part = torch.cos(angle)  # (K, N)
        sin_part = torch.sin(angle)  # (K, N)

        # form real and imag kernels, apply window (only first win_len samples are meaningful)
        real_kernels = cos_part[:, : self.win_len].unsqueeze(1)  # (K,1,win_len)
        imag_kernels = sin_part[:, : self.win_len].unsqueeze(1)  # (K,1,win_len)

        # apply window
        w = self.window.view(1, 1, -1)  # (1,1,win_len)
        real_kernels = real_kernels * w
        imag_kernels = imag_kernels * w

        kernels = torch.cat([real_kernels, imag_kernels], dim=0)  # (2K,1,win_len)
        return kernels  # to be used as conv1d kernels
    
    def forward(self, inputs: torch.Tensor, cplx: bool = True):
        
        B, C, L = inputs.shape
        conv1d_kernel = self._build_kernels()  # (2K,1,win_len)
        pad_size = (self.stride - L % self.stride) % self.stride
        x = inputs.view(B * C, 1, L)
        x = F.pad(x, [0, pad_size])

        # conv1d with NUDFT basis
        outputs = F.conv1d(x, conv1d_kernel.to(x.dtype), stride=self.stride)  # (B*C, 2K, L)
        outputs = outputs.view(B, C, 2 * self.K, -1)  # (B,C,2K,L)
        real, imag = torch.chunk(outputs, 2, dim=2)  # each (B,C,K,L)

        if cplx:
            return real, imag
        else:
            mags = torch.sqrt(real ** 2 + imag ** 2 + EPSILON)
            phase = torch.atan2(imag, real + EPSILON)
            return mags, phase