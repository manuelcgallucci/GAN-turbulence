import torch
import nn_definitions as nn_d

from model_generator import CNNGeneratorBCNocnn1 as CNNGenerator
data_dir = './generated/HenoXL/'

def test_model(n_samples=64, len_=2**15, edge=4096, device="cuda"):
    generator = CNNGenerator().to(device)
    generator.load_state_dict(torch.load(data_dir + 'generator.pt'))
    
    discriminator = nn_d.Discriminator().to(device)
    discriminator.load_state_dict(torch.load(data_dir + 'discriminator.pt'))

    noise = torch.randn((n_samples, 1, len_+2*edge), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)
        generated_samples = generated_samples[:,:,edge:-edge]

        predictions_edge = discriminator(generated_samples)
    acc_edge = torch.sum(predictions_edge < 0.5).item() / n_samples



    noise = torch.randn((n_samples, 1, len_), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)
        generated_samples = generated_samples

        predictions = discriminator(generated_samples)
    acc = torch.sum(predictions < 0.5).item() / n_samples


    print("Discriminator accuracy on the results:")
    print("\tWith edge: {:4.3f}".format(acc_edge))
    print("\t\tMax / Min prediction: {:3.2f} / {:3.2f}".format(torch.max(predictions_edge).item(), torch.min(predictions_edge).item()))
    print("\tNo edge: {:4.3f}".format(acc))
    print("\t\tMax / Min prediction: {:3.2f} / {:3.2f}".format(torch.max(predictions).item(), torch.min(predictions).item()))

if __name__ == "__main__":
    test_model()