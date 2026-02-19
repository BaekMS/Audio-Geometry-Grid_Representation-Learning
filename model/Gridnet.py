import torch
from .util import LayerNorm, fibonacci_sphere, cart2sph

class Gridnet_block(torch.nn.Module):
    def __init__(self, representation_size):
        super(Gridnet_block, self).__init__()

        self.linear = torch.nn.Linear(representation_size, representation_size, bias=True)
        self.activation = torch.nn.ELU()
        self.norm = LayerNorm(representation_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x.unsqueeze(-1)).squeeze(-1)
        return x

class Gridnet(torch.nn.Module):
    def __init__(self, num_blocks=3, representation_size=256):
        super(Gridnet, self).__init__()

        self.num_blocks = num_blocks
        self.blocks = torch.nn.ModuleList()

        for _ in range(self.num_blocks):    
            self.blocks.append(Gridnet_block(
                representation_size=representation_size
            ))        

        self.final_linear = torch.nn.Linear(representation_size, representation_size, bias=True)

    def forward(self, x):   

        for block in self.blocks:
            x = block(x)

        x=self.final_linear(x)

        return x
    
class Gridnet_list(torch.nn.Module):
    def __init__(self, num_depths = 3, num_blocks=3, representation_size=256, fibonacci_size = 2048, xi=1.0):
        super(Gridnet_list, self).__init__()

        self.num_blocks = num_blocks
        self.fibonacci_size = fibonacci_size
        self.representation_size = representation_size
        self.v = torch.arange(0, self.representation_size//4).view(1, -1) * xi
        self.v = torch.nn.Parameter(self.v, requires_grad=False)

        self.gridnet_list = torch.nn.ModuleList()

        for _ in range(self.num_blocks):
            self.gridnet_list.append(Gridnet(
                num_blocks=self.num_blocks,
                representation_size=representation_size
            ))

    def sinusoidal_feature(self, DOA_candidates):
        device = DOA_candidates.device

        azimuth, elevation, distance=cart2sph(DOA_candidates[:,0], DOA_candidates[:,1], DOA_candidates[:,2], is_degree=False)

        azimuth=azimuth.view(-1, 1).to(device)
        elevation=elevation.view(-1, 1).to(device)
        distance=distance.view(-1, 1).to(device)       

        sinusoidal_modulation=[]
        sinusoidal_modulation.append(torch.cos(azimuth + 2*torch.pi * self.v))
        sinusoidal_modulation.append(torch.sin(azimuth + 2*torch.pi * self.v))
        sinusoidal_modulation.append(torch.cos(elevation + 2*torch.pi * self.v))
        sinusoidal_modulation.append(torch.sin(elevation + 2*torch.pi * self.v))

        sinusoidal_modulation=torch.cat(sinusoidal_modulation, dim=-1).to(device)

        spherical_candidates = torch.cat([azimuth, elevation, distance], dim=-1).to(device)      
  
        return sinusoidal_modulation, spherical_candidates

    def forward(self, audio_representaion, DOA_candidates=None):

        if DOA_candidates is None: 
            DOA_candidates = fibonacci_sphere(self.fibonacci_size)  # (fibonacci_size, 3)
        
        DOA_candidates = DOA_candidates.to(audio_representaion.device)

        grid_input,  DOA_spherical_candidates= self.sinusoidal_feature(DOA_candidates)  # (fibonacci_size, representation_size)

        grid_representations = []

        for net in self.gridnet_list:
            grid_representation = net(grid_input)
            grid_representations.append(grid_representation)

        grid_representations = torch.stack(grid_representations, dim=0).unsqueeze(0) 

        G_root = self.representation_size ** 0.5

        spatial_spectrum_output = torch.matmul(grid_representations, audio_representaion/G_root)

        return spatial_spectrum_output.sigmoid(), DOA_candidates, DOA_spherical_candidates