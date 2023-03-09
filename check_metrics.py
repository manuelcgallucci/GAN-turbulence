import os 
import utility as ut
import numpy as np
import json
import torch


from model_generator import CNNGeneratorBigConcat
from model_generator import CNNGeneratorBCNocnn1 


def compute_structures_train(scales, data_path='./data/data.npy', device="cpu"):
    data_train = np.load(data_path)
    data_train = np.flip(data_train, axis=1).copy()
    struct = ut.calculate_structure(torch.Tensor(data_train[:,None,:]), scales, device=device)
    struct_mean_real = torch.mean(struct[:,:,:], dim=0).cpu()
    struct_std_real = torch.std(struct[:,:,:], dim=0).cpu()

    return struct_mean_real, struct_std_real

# has to have the CNNGenerator imported 
def compute_metrics_structures(scales, generator_name, model_dir, struct_mean_real, struct_std_real, edge=4096, len_=2**15, n_samples=128, device="cpu"):
    
    if generator_name == "CNNGeneratorBigConcat":
        generator = CNNGeneratorBigConcat().to(device)
        from model_generator import CNNGeneratorBigConcat as CNNGenerator
    elif generator_name == "CNNGeneratorBCNocnn1":
        generator = CNNGeneratorBCNocnn1().to(device)
    else:
        print("Generator name not defined:", generator_name)
        return None, None, None
    
    try:
        generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pt')))
    except Exception as e:
        print("Error when loading the genrator at:", model_dir)
        return None, None, None

    noise = torch.randn((n_samples, 1, len_+2*edge), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)
    
    generated_samples = generated_samples[:,:,edge:-edge]
    
    struct = ut.calculate_structure(generated_samples, scales, device=device)
    struct_mean_generated = torch.mean(struct[:,:,:], dim=0).cpu()
    struct_std_generate = torch.std(struct[:,:,:], dim=0).cpu()

    # Compute and save structure function metrics
    mse_structure = torch.mean(torch.square(struct_mean_generated - struct_mean_real), dim=1)
    return mse_structure, struct_mean_generated, struct_std_generate

def return_json_dict(name, structure_means, structure_std):
    dic = dict()
    dic[name] = {
        "meanS2": list(structure_means[0,:].tolist()),
        "meanSkew": list(structure_means[1,:].tolist()),
        "meanFlatness": list(structure_means[2,:].tolist()),
        "stdS2": list(structure_std[0,:].tolist()),
        "stdSkew": list(structure_std[1,:].tolist()),
        "stdFlatness": list(structure_std[2,:].tolist())
    }
    return dic

def verify_json_append(json_file, name, struct_mean, struct_std):
    # open the JSON file in read mode to load the existing data
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            f.write("{}")

    with open(json_file, 'r') as f:
        data_dict = json.load(f)

    if not name in data_dict.keys():
        json_dict = return_json_dict(name, struct_mean, struct_std)
        data_dict[name] = json_dict[name]

    with open(json_file, 'w') as f:
        json.dump(data_dict, f)

    return data_dict

def main(device="cuda", json_file="output.json"):
    nv=10
    uu=2**np.arange(0,13,1/nv)
    scales=np.unique(uu.astype(int))
    scales=scales[0:100]

    struct_mean_real, struct_std_real = compute_structures_train(scales, data_path='./data/data.npy', device=device)

    data_dict = verify_json_append(json_file, "data", struct_mean_real, struct_std_real)
     
    models_path = "./generated"
    for model_hash in os.listdir(models_path):
        base_path = os.path.join(models_path, model_hash)

        imported_correctly = False
        if os.path.exists(os.path.join(base_path, "meta.txt")) and os.path.exists(os.path.join(base_path, "generator.pt")):
            
            if not ( model_hash in data_dict.keys()):
                    
                # Search the name of the generator and import it as CNNGenerator
                with open(os.path.join(base_path, "meta.txt"), 'r') as f:
                    for line in f:
                        if 'from model_generator import ' in line and ' as ' in line:
                            start = line.find('from model_generator import ') + len('from model_generator import ')
                            end = line.find(' as ', start)
                            generator_name = line[start:end].replace(' ', '')
                            # import_correct_model(generator_name)
                            print(generator_name)
                            imported_correctly = True
                            break
                    
                # Compute the structure functions and get the metrics
                mse_structure, struct_mean_generated, struct_std_generate = compute_metrics_structures(scales, generator_name, base_path, struct_mean_real, struct_std_real, device=device)
                
                if mse_structure is not None:
                    data_dict[model_hash] = return_json_dict(model_hash, struct_mean_generated, struct_std_generate)[model_hash]
            else:
                print(model_hash, "already had been parsed")  
            imported_correctly = True
        
        if not imported_correctly:
            print(model_hash, "problem with the model")
        else:
            if mse_structure is not None:
                print(model_hash, mse_structure)

    with open(json_file, 'w') as f:
        json.dump(data_dict, f)

    
if __name__ == "__main__":
    main()