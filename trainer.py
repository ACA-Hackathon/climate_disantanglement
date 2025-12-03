import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import DisentangledTripleVAE  # your model file

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 1000
LR = 1e-3
LATENT_DIM = 16
HIDDEN_DIM = 128
SEQ_LEN_METEO = 30  # sequence length for decoder output
SEQ_LEN_PRIX = 20 # sequence length for decoder output
# Input dims
INPUT_DIM_CNN = 10  # e.g., meteo features
INPUT_DIM_RNN = 5   # e.g., price features
INPUT_DIM_MLP = 8   # e.g., soil vector

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

writer = SummaryWriter(log_dir="./runs")

# ==========================================================
# DUMMY DATASET (replace with your real data)
# ==========================================================
# xA: [B, seq_len, F_cnn], xB: [B, seq_len, F_rnn], xC: [B, F_mlp]
num_samples = 1000
xA = torch.randn(num_samples, SEQ_LEN_METEO, INPUT_DIM_CNN)
xB = torch.randn(num_samples, SEQ_LEN_PRIX, INPUT_DIM_RNN)
xC = torch.randn(num_samples, INPUT_DIM_MLP)

dataset = TensorDataset(xA, xB, xC)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================================
# MODEL
# ==========================================================
model = DisentangledTripleVAE(
    input_dim_cnn=INPUT_DIM_CNN,
    input_dim_rnn=INPUT_DIM_RNN,
    input_dim_mlp=INPUT_DIM_MLP,
    seq_len_prix=SEQ_LEN_PRIX,
    seq_len_meteo=SEQ_LEN_METEO,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    num_layers=1
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================
# TRAIN LOOP
# ============================
global_step = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
    running_loss = 0.0

    for batch in loop:
        xA_batch, xB_batch, xC_batch = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()

        outputs, z = model(xA_batch, xB_batch, xC_batch)

        # Compute VAE loss
        mu_list = [outputs["A"]["mu"], outputs["B"]["mu"], outputs["C"]["mu"]]
        logvar_list = [outputs["A"]["logvar"], outputs["B"]["logvar"], outputs["C"]["logvar"]]
        recon = outputs["recon"]

        x = torch.cat([
            xA_batch.view(xA_batch.size(0), -1),
            xB_batch.view(xB_batch.size(0), -1),
            xC_batch
        ], dim=1)

        loss, details = model.vae_loss(recon, x, mu_list, logvar_list)
        # NOTE: xA_batch as target for demo; replace with proper joint target if needed

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        # TensorBoard logging
        writer.add_scalar("train/total_loss", loss.item(), global_step)
        writer.add_scalar("train/recon_loss", details["recon_loss"], global_step)
        writer.add_scalar("train/KLD_A", details["KLD_0"], global_step)
        writer.add_scalar("train/KLD_B", details["KLD_1"], global_step)
        writer.add_scalar("train/KLD_C", details["KLD_2"], global_step)
        global_step += 1

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt"))
    print(f"Epoch {epoch} | Avg Loss: {running_loss / len(loader):.4f}")

writer.close()
