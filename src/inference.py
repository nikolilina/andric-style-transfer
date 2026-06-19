import torch
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


def encode(tokenizer, old_to_new, sentence: str, max_length: int = 128):
    ids = tokenizer(
        str(sentence),
        max_length=max_length,
        truncation=True,
        add_special_tokens=True
    )["input_ids"]
    return [old_to_new.get(i, 0) for i in ids]


def decode(tokenizer, new_to_old, ids: list[int]) -> str:
    original_ids = [new_to_old[i] for i in ids if i in new_to_old]
    return tokenizer.decode(original_ids, skip_special_tokens=True)


def generate(model, tokenizer, old_to_new, new_to_old, sentence: str,
             device: str, max_len: int = 100,
             temperature: float = 0.8, top_k: int = 50):

    src = torch.tensor([encode(tokenizer, old_to_new, sentence)]).to(device)

    # EOS token ID u kompaktnom rečniku
    eos_old = tokenizer.eos_token_id  # = 1 u mT5
    eos_new = old_to_new.get(eos_old, 0)
    bos_new = eos_new  # mT5 nema poseban BOS, koristimo EOS kao start

    generated = [bos_new]

    with torch.no_grad():
        for _ in range(max_len):
            tgt = torch.tensor([generated]).to(device)
            output, _ = model(src, tgt)

            logits = output[0, -1, :]

            logits = torch.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)

            logits = logits / temperature

            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            probs = torch.softmax(top_k_logits, dim=-1)

            probs = torch.clamp(probs, min=1e-9)
            probs = probs / probs.sum()

            next_token = top_k_indices[torch.multinomial(probs, 1)].item()

            if next_token == eos_new:
                break

            generated.append(next_token)

    return decode(tokenizer, new_to_old, generated[1:])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Učitavam mT5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

    model, old_to_new, new_to_old, vocab_size = load_model(
        "data/best_model_mt5.pt", device
    )
    print("Model učitan uspešno!")
    print(f"Veličina rečnika: {vocab_size}")

    test_sentences = [
        "Ljubav izvlači dobre osobine čoveka.",
        "Noć izaziva strah različitog intenziteta.",
        "Sposobnost zaboravljanja meri čovekovu snagu.",
        "Sunce zalazi iza brda.",
        "Dete trči kroz park.",
    ]

    print("\n--- Generisanje ---")
    for sentence in test_sentences:
        result = generate(model, tokenizer, old_to_new, new_to_old,
                         sentence, device)
        print(f"\nUlaz:  {sentence}")
        print(f"Izlaz: {result}")