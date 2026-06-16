import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class AndricDatasetMT5(Dataset):
    def __init__(self, csv_path: str, max_length: int = 128):
        self.df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
        self.pairs = list(zip(self.df["neutral_text"], self.df["original_text"]))
        self.max_length = max_length

        print("Učitavam mT5 tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

        # Remapiranje — gradimo kompaktni rečnik
        self._build_compact_vocab()

    def _build_compact_vocab(self):
        """Skuplja sve token ID-jeve iz dataseta i remapira ih."""
        used_ids = set()
        for neutral, andric in self.pairs:
            for text in [str(neutral), str(andric)]:
                ids = self.tokenizer(
                    text,
                    add_special_tokens=True
                )["input_ids"]
                used_ids.update(ids)

        # Sortirani lista jedinstvenih ID-jeva
        sorted_ids = sorted(used_ids)

        # Mapiranje: stari ID → novi ID
        self.old_to_new = {old: new for new, old in enumerate(sorted_ids)}
        # Mapiranje: novi ID → stari ID (za dekodiranje)
        self.new_to_old = {new: old for old, new in self.old_to_new.items()}

        self.vocab_size = len(sorted_ids)
        print(f"Originalni rečnik mT5: {self.tokenizer.vocab_size}")
        print(f"Kompaktni rečnik (naš dataset): {self.vocab_size}")

    def encode(self, sentence: str) -> list[int]:
        """Enkoduje rečenicu i primenjuje remapiranje."""
        original_ids = self.tokenizer(
            str(sentence),
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True
        )["input_ids"]
        # Nepoznate tokene mapiramo na 0 (UNK)
        return [self.old_to_new.get(i, 0) for i in original_ids]

    def decode(self, new_ids: list[int]) -> str:
        """Primenjuje obrnuto remapiranje i dekoduje."""
        original_ids = [self.new_to_old[i] for i in new_ids
                       if i in self.new_to_old]
        return self.tokenizer.decode(original_ids, skip_special_tokens=True)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        neutral, andric = self.pairs[idx]
        return self.encode(neutral), self.encode(andric)


# ── Brza provjera ──
if __name__ == "__main__":
    dataset = AndricDatasetMT5(
        "data/andric_style_transfer_dataset101.csv"
    )

    print(f"Broj parova: {len(dataset)}")

    test = "Pada kiša na grad."
    encoded = dataset.encode(test)
    decoded = dataset.decode(encoded)

    print(f"\nOriginalna rečenica: {test}")
    print(f"Tokeni (novi ID-jevi): {encoded}")
    print(f"Dekodirana rečenica:   {decoded}")

    neutral, andric = dataset[0]
    print(f"\nPrvi par:")
    print(f"  Neutral: {neutral}")
    print(f"  Andrić:  {andric}")