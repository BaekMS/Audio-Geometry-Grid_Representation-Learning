import torch
from .input_feature_process import input_feature_process
from .LNuDFT import LearnableNuDFT
from .Channel_invariant_feature_extractor import Channel_invariant_feature_extractor
from .Microphone_positional_encoding import MicrophonePositionalEncoding
from .SpatioTemporal_block import SpatioTemporal_block
from .Representation_mapping import Rpresentation_mapping_block
from .Gridnet import Gridnet_list
from .util import target_spatial_spectrum

class AGG_RL(torch.nn.Module):
    def __init__(self, MPE_type):
        super(AGG_RL, self).__init__()

        win_len = 512
        feature_size = 128
        ref_channel = 0
        MPE_type = MPE_type
        representation_size = 256
        fibonacci_size = 2048
        self.ref_channel=ref_channel

        self.input_feature_process=input_feature_process(win_len=win_len, ref_channel=ref_channel)

        self.MPE = MicrophonePositionalEncoding(feature=feature_size,
                                                MPE_type=MPE_type,
                                                alpha = 7,
                                                beta = 4,
                                                ref_channel=ref_channel,)

        self.CIFE=Channel_invariant_feature_extractor(init_feature = win_len+2,
                                                      feature = feature_size, 
                                                      kernel_size = 3, 
                                                      padding = 1, 
                                                      stride = 1, 
                                                      dilation_rate = 2, 
                                                      num_blocks = 4,)

        self.MPE = MicrophonePositionalEncoding(feature=feature_size,
                                                MPE_type=MPE_type,
                                                alpha = 7,
                                                beta = 4)        

        self.STDPBs=SpatioTemporal_block(n_heads=16,
                                        num_blocks=4,
                                        feature_size=feature_size,
                                        rnn_layers=2,)        
        
        self.RMBs=Rpresentation_mapping_block(feature = feature_size,
                                            representation_size = representation_size,
                                            num_blocks = 3,
                                            kernel_size = 3, 
                                            dilation_rate = 2)      

        self.gridnet = Gridnet_list(num_depths=3, num_blocks=3, representation_size=representation_size, fibonacci_size =fibonacci_size, xi=1.0) 

        self.gammas = [5.0, 5.0, 5.0]

    def ref_channel_modify_to_center(self, x, mic_coordinate):
        
        center = mic_coordinate.mean(dim=1, keepdim=True)  # B x 1 x 3      

        nearest_channel = torch.argmin(torch.norm(mic_coordinate - center, dim=-1), dim=1)

        x_new = x.clone()
        mic_coordinate_new = mic_coordinate.clone()
        rc = self.ref_channel 

        # Gather values to swap
        batch_indices = torch.arange(x.shape[0], device=x.device)
        # Swap x
        x_new[batch_indices, rc], x_new[batch_indices, nearest_channel] = \
            x[batch_indices, nearest_channel].clone(), x[batch_indices, rc].clone()
        # Swap mic_pos
        mic_coordinate_new[batch_indices, rc], mic_coordinate_new[batch_indices, nearest_channel] = \
            mic_coordinate[batch_indices, nearest_channel].clone(), mic_coordinate[batch_indices, rc].clone()  
        
        return x_new, mic_coordinate_new
    
    def forward(self, x, mic_coordinate, vad=None, target_spherical_position=None, return_target=False):
        x=x.contiguous().to(self.MPE.v.device)

        # Reference channel modification
        x, mic_coordinate=self.ref_channel_modify_to_center(x, mic_coordinate)
        
        # GCC-PHAT feature
        x_feature = self.input_feature_process(x)
        x_feature=torch.cat(x_feature, dim=2) # B, C-1, 2F, L


        # Microphone coordinates processing
        MPE, mic_coord_cart, mic_coord_dist_sin_cos = self.MPE(mic_coordinate) 
        
        # Channel invariant feature extraction
        x_feature=self.CIFE(x_feature, MPE, mic_coord_cart, mic_coord_dist_sin_cos) # B, C-1, M, L

        # Spatio-temporal dual-path block
        x_spatio_temporal=self.STDPBs(x_feature, MPE) # B, C-1, M, T 

        # Audio Representation mapping
        audio_representaion = self.RMBs(x_spatio_temporal)  # B, DS, G, L

        # Probability spatial spectrum estimation
        x_out, DOA_cart_candidates, DOA_spherical_candidates=self.gridnet(audio_representaion)  # B, DS, D, L
        
        if return_target:
            vad_framed = self.input_feature_process.LNuDFT.get_vad_framed(vad)             

            target = target_spatial_spectrum(target_spherical_position, DOA_cart_candidates, vad_framed, self.gammas, self.input_feature_process.LNuDFT.get_trajectory_framed)  # B, DS, D, L

            return x_out, target, DOA_cart_candidates, DOA_spherical_candidates
        
        else:
            return x_out
