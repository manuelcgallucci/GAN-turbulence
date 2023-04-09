import torch
from model_generator import MODEL as CNNGenerator 

data_dir = './generated/LdovFU/'

def get_int_outs(n_samples=1, len_=2**15, edge=4096, device="cuda"):
    
    generator = CNNGenerator().to(device)
    generator.load_state_dict(torch.load(data_dir + 'generator.pt'))
    noise = torch.randn((n_samples, 1, len_+2*edge), device=device)
    with torch.no_grad():
        out_res = generator.get_inter(noise)
    # generated_samples = generated_samples[:,:,edge:-edge]
    
    for output in out_res:
        print(output.shape)
        


if __name__ == "__main__":
    get_int_outs()