import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Transformer
from transformers import T5Tokenizer


def load_model(model_path: str, device: str):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    old_to_new = checkpoint["old_to_new"]
    new_to_old = checkpoint["new_to_old"]
    vocab_size  = checkpoint["vocab_size"]
    model = Transformer(vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, old_to_new, new_to_old, vocab_size


def encode(tokenizer, old_to_new, sentence: str):
    ids = tokenizer(
        str(sentence),
        add_special_tokens=True
    )["input_ids"]
    return [old_to_new.get(i, 0) for i in ids]


def decode_token(tokenizer, new_to_old, token_id: int) -> str:
    old_id = new_to_old.get(token_id, 0)
    return tokenizer.convert_ids_to_tokens([old_id])[0]


def get_attention_weights(model, tokenizer, old_to_new, new_to_old,
                          sentence: str, device: str, max_len: int = 30):
    """Generišemo izlaz i hvatamo attention težine iz poslednjeg sloja dekodera."""

    src_ids = encode(tokenizer, old_to_new, sentence)
    src = torch.tensor([src_ids]).to(device)

    eos_old = tokenizer.eos_token_id  # = 1 u mT5
    eos_new = old_to_new.get(eos_old, -1)

    # Koristimo poseban start token — uzimamo prvi token iz dataseta
    bos_new = 1  # bilo koji ne-EOS, ne-PAD token
    generated = [bos_new]

    all_cross_attention = []

    with torch.no_grad():
        for _ in range(max_len):
            tgt = torch.tensor([generated]).to(device)

            # Forward pass
            output, _ = model(src, tgt)

            # Dohvati cross-attention iz poslednjeg decoder sloja
            last_decoder_layer = model.decoder.layers[-1]
            cross_attn = last_decoder_layer.cross_attention

            # Registruj hook za hvatanje attention težina
            attn_weights = []
            def hook(module, input, output):
                # output je (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) == 2:
                    attn_weights.append(output[1])

            handle = cross_attn.register_forward_hook(hook)

            # Ponovi forward da uhvatimo težine
            _, _ = model(src, tgt)
            handle.remove()

            logits = output[0, -1, :]
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)
            next_token = torch.argmax(logits).item()

            #if next_token == eos_new or next_token == 0:
            #   break
            if next_token == 0:  # samo PAD zaustavlja
                break

            generated.append(next_token)

            if attn_weights:
                all_cross_attention.append(attn_weights[0])

    return src_ids, generated[1:], all_cross_attention


def plot_attention(tokenizer, old_to_new, new_to_old,
                   src_ids, tgt_ids, attention_list, sentence: str):
    """Crta heatmapu cross-attention težina."""

    if not attention_list:
        print("Nema attention težina za prikaz.")
        return

    # Dekoduj tokene za labele
    src_tokens = [decode_token(tokenizer, new_to_old, i) for i in src_ids]
    tgt_tokens = [decode_token(tokenizer, new_to_old, i) for i in tgt_ids]

    # Skupi attention matrice — shape: (tgt_len, src_len)
    attn_matrix = []
    for attn in attention_list:
        # attn shape: (batch, heads, tgt_seq, src_seq)
        # Uzimamo prosek po glavama, poslednji tgt token
        if attn is not None:
            avg = attn[0].mean(dim=0)  # (tgt_seq, src_seq)
            attn_matrix.append(avg[-1, :len(src_ids)].cpu().numpy())

    if not attn_matrix:
        print("Attention matrica je prazna.")
        return

    attn_matrix = np.array(attn_matrix)  # (tgt_len, src_len)

    # Ograniči dužine za čitljivost
    max_tgt = min(len(tgt_tokens), attn_matrix.shape[0])
    max_src = min(len(src_tokens), attn_matrix.shape[1])
    attn_matrix = attn_matrix[:max_tgt, :max_src]
    tgt_tokens = tgt_tokens[:max_tgt]
    src_tokens = src_tokens[:max_src]

    fig, ax = plt.subplots(figsize=(max(8, max_src), max(6, max_tgt // 2)))
    im = ax.imshow(attn_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(tgt_tokens, fontsize=10)

    ax.set_xlabel("Ulazni tokeni (neutralna rečenica)", fontsize=12)
    ax.set_ylabel("Izlazni tokeni (Andrićev stil)", fontsize=12)
    ax.set_title(f"Cross-Attention težine\nUlaz: '{sentence}'", fontsize=13)

    plt.colorbar(im, ax=ax, label="Težina pažnje")
    plt.tight_layout()

    safe_sentence = sentence[:20].replace(" ", "_").replace(".", "")
    output_path = f"data/attention_{safe_sentence}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Vizualizacija sačuvana: {output_path}")
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Učitavam mT5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

    model, old_to_new, new_to_old, vocab_size = load_model(
        "data/best_model_mt5.pt", device
    )
    print(f"Model učitan! Veličina rečnika: {vocab_size}")

    test_sentences = [
        "Ljubav izvlači dobre osobine čoveka.",
        "Noć izaziva strah različitog intenziteta.",
        "Sposobnost zaboravljanja meri čovekovu snagu.",
        "Sunce zalazi iza brda.",
        "Dete trči kroz park.",
    ]

    for sentence in test_sentences:
        print(f"\nGenerišem za: '{sentence}'")
        src_ids, tgt_ids, attention_list = get_attention_weights(
            model, tokenizer, old_to_new, new_to_old, sentence, device
        )

        if tgt_ids:
            plot_attention(
                tokenizer, old_to_new, new_to_old,
                src_ids, tgt_ids, attention_list, sentence
            )
        else:
            print("Model nije generisao nijedan token.")