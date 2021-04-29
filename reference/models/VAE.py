import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Variational Autoencoder
class VAE(nn.Module):
	def __init__(self, encoder, decoder):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x):
		# Encoder
		z_mu, z_var = self.encoder(x)

		# Re-parameterization trick: Sample from the distribution having latent parameters z_mu, z_var
		std = torch.exp(z_var / 2)
		eps = torch.randn_like(std)
		x_sample = eps.mul(std).add_(z_mu)

		# Decoder
		predict = self.decoder(x_sample)
		return predict, z_mu, z_var

'''
Usage:

# Encoder
encoder = Encoder(...) # This is nn.Module

# Decoder
decoder = Decoder(...) # This is nn.Module

# VAE
model = VAE(encoder, decoder).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = <Learning rate>)

# sample and generate a image
z = torch.randn(1, <Latent dimension>).to(device)

# run only the decoder
reconstructed = model.dec(z)
'''