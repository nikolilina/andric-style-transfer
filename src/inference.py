import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Transformer


def load_model(model_path: str, device: str):
    from src.dataset import Vocabulary
    torch.serialization.add_safe_globals([Vocabulary])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]

    model = Transformer(vocab_size=len(vocab.word2idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab


def generate(model, vocab, sentence: str, device: str,
             max_len: int = 100, temperature: float = 0.8, top_k: int = 50):
    # Enkoduj ulaznu rečenicu
    src = torch.tensor([vocab.encode(sentence)]).to(device)

    # Počni generisanje sa BOS tokenom
    generated = [2]  # BOS_IDX

    with torch.no_grad():
        for _ in range(max_len):
            tgt = torch.tensor([generated]).to(device)
            output = model(src, tgt)

            # Uzmi logite za poslednju poziciju
            logits = output[0, -1, :] / temperature

            # Top-K sampling
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[torch.multinomial(probs, 1)].item()

            # Stani ako je EOS
            if next_token == 3:  # EOS_IDX
                break

            generated.append(next_token)

    # Pretvori indekse nazad u reči
    words = [vocab.idx2word.get(idx, "[UNK]") for idx in generated[1:]]
    return " ".join(words)


# ── Brza provjera ──
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, vocab = load_model("data/best_model.pt", device)
    print("Model učitan uspešno!")
    print(f"Veličina rečnika: {len(vocab.word2idx)}")

    # Testiraj sa nekoliko rečenica
    test_sentences = [
        "Pada kiša na grad.",
        "Čovek hoda ulicom.",
        "Sunce zalazi iza brda.",
    ]

    print("\n--- Generisanje ---")
    for sentence in test_sentences:
        result = generate(model, vocab, sentence, device)
        print(f"\nUlaz:  {sentence}")
        print(f"Izlaz: {result}")