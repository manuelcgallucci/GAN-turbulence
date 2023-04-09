import torch
from model_generator import CNNGeneratorBigConcat as CNNGenerator
from model_discriminator import DiscriminatorMultiNet16_4 as Discriminator
from model_discriminator import DiscriminatorStructures_v2 as DiscriminatorStructures


def count_params(model):
	trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
	)
	return trainable_params



device="cpu"



generator = CNNGenerator().to(device)
print("Generator:", count_params(generator))

discriminator = Discriminator().to(device)
print("Discriminator:", count_params(discriminator))

discriminator_structures = DiscriminatorStructures().to(device)
print("Discriminator structures:", count_params(discriminator_structures))

