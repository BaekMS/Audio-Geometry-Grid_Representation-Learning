import torch
from model.main import AGG_RL
from glob import glob
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def plot_molleweide(ax, azimuth_candidate, elevation_candidate, data):
    data = data.mean(axis=-1)    

    ax.grid(True)
    ax.grid(True, linestyle='--', linewidth=1)
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_dashes([10, 10])  
    ax.set_position([0.1, 0.1, 0.8, 0.8])  

    for spine in ax.spines.values():
        spine.set_visible(False)

    elevation_candidate = -elevation_candidate + np.pi/2

    sc = ax.scatter(azimuth_candidate, elevation_candidate, c=data, s=150, cmap='viridis', vmin=0, vmax=1, marker='8')

    ax.tick_params(axis='x', colors='white')
    ax.set_xlabel('Azimuth')
    ax.set_xticks([ -np.pi/3*2, -np.pi/3, 0, np.pi/3, np.pi/3*2])
    ax.set_xticklabels(['-120°', '-60°', '0°', '60°', '120°'])

    ax.set_ylabel('Elevation')
    ax.yaxis.set_label_position('left')   
    ax.yaxis.tick_right()                 
    ax.set_yticks([-np.pi/3, 0, np.pi/3])
    ax.set_yticklabels(['120°', '90°', '60°'])      

    ax.tick_params(axis='x', length=0)  
    ax.tick_params(axis='y', length=0)  
    plt.colorbar(sc, ax=ax)  

def plot_spectrum(output, target, file_name, DOA_cart_candidates, DOA_spherical_candidates):
    png_dir = './spectrum_plots/' + file_name

    output = output.cpu().numpy()[0]
    target = target.cpu().numpy()[0]

    main_figure = plt.figure(figsize=(12, 9))
    plt.rcParams['font.size'] = 7
    num_layers = output.shape[0]
    
    for i in range(num_layers):
        ax=main_figure.add_subplot(num_layers, 2, 2*i + 1, projection="mollweide")
        plot_molleweide(ax, DOA_spherical_candidates[:,0], DOA_spherical_candidates[:,1], output[i])
        ax.set_title('Layer {}, Predict'.format(i+1))        

        ax=main_figure.add_subplot(num_layers, 2, 2*i + 2, projection="mollweide")
        plot_molleweide(ax, DOA_spherical_candidates[:,0], DOA_spherical_candidates[:,1], target[i])
        ax.set_title('Layer {}, Target'.format(i+1))

    plt.savefig(png_dir, bbox_inches='tight', dpi=100, pad_inches=0.1)
    plt.close()
    plt.cla()
    plt.clf()

def to_tensor(data, device):
    data = data.to_numpy()
    data = np.stack(data, axis=0)  
    data = torch.from_numpy(data).to(device).float()  

    return data
    

def inference(model, device):
    # Load the data paths
    data_list = glob('./sample_input/**/*.pkl', recursive=True) 

    with torch.no_grad():
        model.eval()

        for data_path in tqdm(data_list, total=len(data_list), desc='Inference'):
            with open(data_path, 'rb') as f:
                data = pkl.load(f)

            # Unpack data
            input_audio = to_tensor(data['input_audio'], device)  
            vad = to_tensor(data['vad'], device)
            mic_coordinate = to_tensor(data['mic_coordinate'], device)
            spherical_position = to_tensor(data['spherical_position'], device) 

            # Forward pass through the model
            pred, target, DOA_cart_candidates, DOA_spherical_candidates = model(
                input_audio,
                mic_coordinate,
                vad=vad,
                target_spherical_position=spherical_position,
                return_target=True         
            )
            DOA_cart_candidates = DOA_cart_candidates.cpu().numpy()
            DOA_spherical_candidates = DOA_spherical_candidates.cpu().numpy()
            
            data_path = data_path.split('/')
            png_name = data_path[-2] + 'ch_' + data_path[-1].replace('.pkl', '.png')

            plot_spectrum(pred, target, png_name, DOA_cart_candidates, DOA_spherical_candidates)
            

if __name__ == "__main__":    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained = torch.load('./pretrained/AGG_RL_pretrained.tar', map_location='cpu')
    model = AGG_RL(MPE_type='FM').to(device)
    model.load_state_dict(pretrained['model_state_dict'], strict=True)  
    model.to(device)
    inference(model, device)

  