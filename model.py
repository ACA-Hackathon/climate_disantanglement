import torch
from torch import nn
from torch.nn import functional as F


class CNNEncoder(nn.Module):
    """Simple 1D CNN encoder producing mu and logvar."""
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 latent_dim,
                 kernel_size=3,
                 stride=2,
                 num_layers=2):
        super().__init__()

        layers = []
        in_channels = input_dim

        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels,
                                    hidden_dim,
                                    kernel_size,
                                    stride=stride,
                                    padding=kernel_size//2))
            layers.append(nn.ReLU())
            in_channels = hidden_dim

        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: [B, S, F] → [B, F, S]
        x = x.permute(0, 2, 1)
        h = self.cnn(x)
        h = self.pool(h).squeeze(-1)  # [B, hidden_dim]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class RNNEncoder(nn.Module):
    """Simple unidirectional RNN encoder producing mu and logvar."""
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, h_last = self.rnn(x)
        h_last = h_last[-1]  # [B, hidden_dim]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar


class VectorMLPEncoder(nn.Module):
    """
    takes a vector x
    outputs mu and logvar
    """
    def __init__(self,
                 input_dim,    # size of input vector
                 latent_dim,   # latent dimension
                 hidden_dims=[256, 128, 128]):
        super().__init__()

        layers = []
        in_dim = input_dim

        # build hidden MLP layers
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.mlp = nn.Sequential(*layers)

        # output layers
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x):
        """
        x: [batch, input_dim]
        """
        h = self.mlp(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class MLPDecoder(nn.Module):
    """Maps latent -> flattened sequence -> reshape back."""
    def __init__(self,
                 latent_dim,
                 output_dim_decoder,
                 hidden_dims=[128, 128]):
        super().__init__()

        layers = []
        in_dim = latent_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        
        layers.append(nn.Linear(in_dim, output_dim_decoder))
        self.mlp = nn.Sequential(*layers)

        self.output_dim = output_dim_decoder
    def forward(self, z):
        out = self.mlp(z)
        return out  # [B, meteo_len * meteo_vars * prix_len * prix_vars * soil_dim]


class DisentangledTripleVAE(nn.Module):
    def __init__(self,
                 input_dim_cnn,
                 input_dim_rnn,
                 input_dim_mlp,
                 seq_len_prix,
                 seq_len_meteo,
                 hidden_dim,
                 latent_dim,
                 num_layers=2):
        super().__init__()

        # Encoders
        self.encoder_meteo = CNNEncoder(input_dim_cnn, hidden_dim, latent_dim, num_layers)
        self.encoder_prix = RNNEncoder(input_dim_rnn, hidden_dim, latent_dim, num_layers)
        self.encoder_sol = VectorMLPEncoder(input_dim_mlp, latent_dim)
        self.output_dim_decoder = seq_len_meteo * input_dim_cnn + seq_len_prix * input_dim_rnn + input_dim_mlp
        # Shared decoder takes concatenated latent vector
        self.latent_dim_decoder = latent_dim * 3
        self.decoder = MLPDecoder(
            latent_dim=self.latent_dim_decoder,
            output_dim_decoder=self.output_dim_decoder,
            hidden_dims=[128, 128]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, xA, xB, xC):
        # Encode
        mu_A, log_A = self.encoder_meteo(xA)
        mu_B, log_B = self.encoder_prix(xB)
        mu_C, log_C = self.encoder_sol(xC)

        # Reparameterize
        z_A = self.reparameterize(mu_A, log_A)
        z_B = self.reparameterize(mu_B, log_B)
        z_C = self.reparameterize(mu_C, log_C)

        # Concatenate all latents
        z = torch.cat([z_A, z_B, z_C], dim=-1)

        # Decode concatenated latent
        recon = self.decoder(z)

        # Return per-encoder info + joint reconstruction
        outputs = {
            "A": {"mu": mu_A, "logvar": log_A, "z": z_A},
            "B": {"mu": mu_B, "logvar": log_B, "z": z_B},
            "C": {"mu": mu_C, "logvar": log_C, "z": z_C},
            "recon": recon
        }

        return outputs, z

    # -------------------------------------------------------
    def vae_loss(self,
                 recon,
                 x_target,
                 mu_list,
                 logvar_list,
                 kld_weight=1.0):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x_target)

        kld_total = 0.0
        kld_details = {}
        for i, (mu, logvar) in enumerate(zip(mu_list, logvar_list)):
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kld_total += kld
            kld_details[f"KLD_{i}"] = kld.detach()

        total_loss = recon_loss + kld_weight * kld_total

        details = {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            **kld_details
        }
        return total_loss, details
