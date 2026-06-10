import pandas as pd
from torch.utils.data import Dataset

# ── Specijalni tokeni ──
PAD_IDX = 0  # padding
UNK_IDX = 1  # nepoznata reč
BOS_IDX = 2  # početak rečenice
EOS_IDX = 3  # kraj rečenice


class Vocabulary:
    def __init__(self):
        self.word2idx = {
            "[PAD]": PAD_IDX,
            "[UNK]": UNK_IDX,
            "[BOS]": BOS_IDX,
            "[EOS]": EOS_IDX,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_idx = 4

    def build(self, sentences: list[str]) -> None:
        for sentence in sentences:
            for word in sentence.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.next_idx
                    self.idx2word[self.next_idx] = word
                    self.next_idx += 1

    def encode(self, sentence: str) -> list[int]:
        tokens = [BOS_IDX]
        for word in sentence.lower().split():
            tokens.append(self.word2idx.get(word, UNK_IDX))
        tokens.append(EOS_IDX)
        return tokens

    def __len__(self) -> int:
        return len(self.word2idx)


class AndricDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
        self.pairs = list(zip(df["neutral_text"], df["original_text"]))

        self.vocab = Vocabulary()
        all_sentences = [s for pair in self.pairs for s in pair]
        self.vocab.build(all_sentences)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        neutral, andric = self.pairs[idx]
        return (
            self.vocab.encode(neutral),
            self.vocab.encode(andric),
        )


# ── Brza provjera ──────────────────────────────────────────
if __name__ == "__main__":
    dataset = AndricDataset("data/andric_style_transfer_dataset101.csv")

    print(f"Broj parova: {len(dataset)}")
    print(f"Veličina rečnika: {len(dataset.vocab)}")

    neutral, andric = dataset[0]
    print(f"\nPrvi par:")
    print(f"  Neutral (brojevi): {neutral}")
    print(f"  Andrić  (brojevi): {andric}")