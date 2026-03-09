import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiVAE(nn.Module):
    def __init__(self, item_count, latent_dim, encoder_dims, decoder_dims, beta=0.2, dropout=0.5):
        super().__init__()

        self.item_count = item_count
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # input data was dropped out in paper

        self.dropout = nn.Dropout(dropout)

        # item_count -> encoder dims -> latent_dim*2  ( Encoder )

        encoder_layers = []
        now_dim = self.item_count

        for next_dim in encoder_dims:
            encoder_layers.append(nn.Linear(now_dim, next_dim))
            encoder_layers.append(nn.ReLU())
            now_dim = next_dim
        encoder_layers.append(nn.Linear(now_dim, latent_dim*2))

        self.encoder = nn.Sequential(*encoder_layers)

        # latent_dim -> decoder dims -> item_count ( Decoder )
        # Before inputting vectors to decoder, data must be through reparameterization function

        decoder_layers = []
        now_dim = self.latent_dim

        for next_dim in decoder_dims:
            decoder_layers.append(nn.Linear(now_dim, next_dim))
            decoder_layers.append(nn.ReLU())
            now_dim = next_dim
        decoder_layers.append(nn.Linear(now_dim, item_count))
        
        self.decoder = nn.Sequential(*decoder_layers)
        # After returned from decoder, result must be through softmax fucntion

    def reparameterize(self, mu, logvar):

        # sampling stage, noted as q in paper
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        
        z = mu + std * eps
        return z

    def forward(self, x):
        
        x = self.dropout(x)

        inst = self.encoder(x)
        mu, logvar = torch.chunk(inst, chunks=2, dim=1)

        z = self.reparameterize(mu, logvar)

        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # loss is formed two parts (reconstuct loss & KL Divergence)
        log_prob =  F.log_softmax(recon_x, dim=1)
        recon_loss = -torch.sum(x * log_prob, dim=1).mean()

        kl_loss = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - 1 - logvar, dim=1).mean()

        return recon_loss + self.beta * kl_loss