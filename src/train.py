import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_mt5 import AndricDatasetMT5
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
    "patience": 30,
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
dataset = AndricDatasetMT5(CONFIG["data_path"])

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    collate_fn=collate_fn
)

print(f"Train parova: {train_size} | Val parova: {val_size}")
print(f"Broj parova: {len(dataset)}")
print(f"Broj batcha po epohi: {len(dataloader)}")
print(f"Veličina rečnika: {dataset.vocab_size}")


# ── Model, optimizer, loss ──
model = Transformer(
    vocab_size=dataset.vocab_size,
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


# ── Signal handler za Ctrl+C ──
loss_history = []
val_loss_history = []

def handle_interrupt(sig, frame):
    print("\nPrekid detektovan — crtam graf i izlazim...")
    if loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history,
                 color="steelblue", linewidth=2, label="Train loss")
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history,
                 color="tomato", linewidth=2, label="Val loss")
        plt.title("Loss kroz epohe — mT5, 550 parova, early stopping", fontsize=14)
        plt.xlabel("Epoha")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("data/loss_mt5_val.png", dpi=150)
        print("Graf sačuvan: data/loss_mt5_val.png")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)


# ── Trening petlja ──
best_loss = float("inf")
patience_counter = 0

for epoch in range(CONFIG["num_epochs"]):
    # ── Trening ──
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src = src.to(CONFIG["device"])
        tgt = tgt.to(CONFIG["device"])

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output = model(src, tgt_input)
        output = output.reshape(-1, dataset.vocab_size)
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)

    # ── Validacija ──
    model.eval()
    val_total = 0
    with torch.no_grad():
        for src, tgt in val_dataloader:
            src = src.to(CONFIG["device"])
            tgt = tgt.to(CONFIG["device"])
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            output = output.reshape(-1, dataset.vocab_size)
            tgt_output = tgt_output.reshape(-1)
            val_loss = criterion(output, tgt_output)
            val_total += val_loss.item()

    avg_val_loss = val_total / len(val_dataloader)
    val_loss_history.append(avg_val_loss)

    # ── Early stopping na osnovu val lossa ──
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_loss,
            "old_to_new": dataset.old_to_new,
            "new_to_old": dataset.new_to_old,
            "vocab_size": dataset.vocab_size,
        }, "data/best_model_mt5.pt")
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoha {epoch+1}/{CONFIG['num_epochs']} | "
              f"Train: {avg_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"Patience: {patience_counter}/{CONFIG['patience']}")

    if patience_counter >= CONFIG["patience"]:
        print(f"\nEarly stopping — val loss se nije poboljšao "
              f"{CONFIG['patience']} epoha.")
        print(f"Trening zaustavljen u epohi {epoch+1}.")
        break

print(f"Trening završen! Najbolji val loss: {best_loss:.4f}")

# ── Graf lossa ──
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history,
         color="steelblue", linewidth=2, label="Train loss")
plt.plot(range(1, len(val_loss_history) + 1), val_loss_history,
         color="tomato", linewidth=2, label="Val loss")
plt.title("Loss kroz epohe — mT5, 550 parova, early stopping", fontsize=14)
plt.xlabel("Epoha")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/loss_mt5_val.png", dpi=150)
print("Graf sačuvan: data/loss_mt5_val.png")