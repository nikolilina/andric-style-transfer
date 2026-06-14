import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import AndricDataset
from src.model import Transformer

# ── Konfiguracija ──
CONFIG = {
    "data_path": "data/andric_style_transfer_dataset101.csv",
    "batch_size": 16,
    "num_epochs": 300,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "d_model": 256,
    "num_heads": 4,
    "d_ff": 1024,
    "num_layers": 3,
    "dropout": 0.1,
    "patience": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"Koristimo uređaj: {CONFIG['device']}")


# ── Collate funkcija ──
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(
        [torch.tensor(s) for s in src_batch],
        batch_first=True,
        padding_value=0
    )
    tgt_padded = pad_sequence(
        [torch.tensor(t) for t in tgt_batch],
        batch_first=True,
        padding_value=0
    )
    return src_padded, tgt_padded


# ── Učitavanje podataka ──
dataset = AndricDataset(CONFIG["data_path"])
dataloader = DataLoader(
    dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

print(f"Broj parova: {len(dataset)}")
print(f"Broj batcha po epohi: {len(dataloader)}")
print(f"Veličina rečnika: {len(dataset.vocab)}")


# ── Model, optimizer, loss ──
model = Transformer(
    vocab_size=len(dataset.vocab),
    d_model=CONFIG["d_model"],
    num_heads=CONFIG["num_heads"],
    d_ff=CONFIG["d_ff"],
    num_layers=CONFIG["num_layers"],
    dropout=CONFIG["dropout"],
).to(CONFIG["device"])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)
criterion = nn.CrossEntropyLoss(ignore_index=0)


# ── Trening petlja ──
best_loss = float("inf")
loss_history = []
patience_counter = 0

for epoch in range(CONFIG["num_epochs"]):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src = src.to(CONFIG["device"])
        tgt = tgt.to(CONFIG["device"])

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output = model(src, tgt_input)
        output = output.reshape(-1, len(dataset.vocab))
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_loss,
            "vocab": dataset.vocab,
        }, "data/best_model_es.pt")
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoha {epoch+1}/{CONFIG['num_epochs']} | Loss: {avg_loss:.4f} | Patience: {patience_counter}/{CONFIG['patience']}")

    if patience_counter >= CONFIG["patience"]:
        print(f"\nEarly stopping — loss se nije poboljšao {CONFIG['patience']} epoha.")
        print(f"Trening zaustavljen u epohi {epoch+1}.")
        break

print(f"Trening završen! Najbolji loss: {best_loss:.4f}")


# ── Graf lossa ──
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, color="steelblue", linewidth=2)
plt.title("Loss kroz epohe — Early Stopping (101 par)", fontsize=14)
plt.xlabel("Epoha")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/loss_early_stopping.png", dpi=150)
plt.show()
print("Graf sačuvan: data/loss_early_stopping.png")