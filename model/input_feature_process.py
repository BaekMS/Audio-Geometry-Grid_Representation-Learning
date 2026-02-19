import torch
from .LNuDFT import LearnableNuDFT

class input_feature_process(torch.nn.Module):
    def __init__(self, win_len, ref_channel=0):
        super(input_feature_process, self).__init__()
        self.LNuDFT=LearnableNuDFT(
                            win_len = win_len,
                            hop_len = 128,
                            K = 257, 
                            vad_threshold = 2/3,
                            win_type = 'hann',
                            fs = 16000.0,
                            a_k_trainable = True,
                            eps_0 = 0.01,
                            eps_1 = 100,
                            nu_0 = 0.0,
                            eps_start = 0.15,
                            eps_end = 0.95,
                        )

        self.ref_channel=ref_channel
        self.eps=1e-8    
    
    def cross_correlation(self, x_real, x_imag, PHAT=True):
        comp=torch.complex(x_real, x_imag)

        if PHAT:
            comp = torch.div(comp, torch.clamp(comp.abs(), min=self.eps))
        comp_ref=comp[:, [self.ref_channel], :, :]
        comp = torch.cat([comp[:, :self.ref_channel, :], comp[:, self.ref_channel + 1:, :]], dim=1)
        comp = comp * comp_ref.conj() 
        return comp.real, comp.imag

    def forward(self, x):

        x_LNuDFT_real, x_LNuDFT_imag=self.LNuDFT(x)  # B x C x 2F x T
        GCC_real, GCC_imag=self.cross_correlation(x_LNuDFT_real, x_LNuDFT_imag, PHAT=True)       
        
        return GCC_real, GCC_imag