import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
import os

# ── Specijalni tokeni ──
# SentencePiece automatski rezerviše:
# 0 = <unk>, 1 = <s> (BOS), 2 = </s> (EOS)
# Mi dodajemo PAD kao 3
PAD_IDX = 3


def train_sentencepiece(csv_path: str, model_prefix: str = "data/spm_andric",
                        vocab_size: int = 500):
    """
    Trenira SentencePiece model na svim rečenicama iz dataseta.
    vocab_size = 500 je razuman izbor za dataset od ~101 para.
    """
    # Učitaj sve rečenice u privremeni txt fajl
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    all_sentences = (
        df["original_text"].tolist() + df["neutral_text"].tolist()
    )

    temp_path = "data/spm_temp_corpus.txt"
    with open(temp_path, "w", encoding="utf-8") as f:
        for sentence in all_sentences:
            f.write(str(sentence).strip() + "\n")

    # Treniraj SentencePiece model
    spm.SentencePieceTrainer.train(
        input=temp_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        pad_id=PAD_IDX,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_piece="[PAD]",
        character_coverage=1.0,
        normalization_rule_name="identity",
        input_sentence_size=10000,
        shuffle_input_sentence=True,
    )

    # Obriši privremeni fajl
    os.remove(temp_path)
    print(f"SentencePiece model sačuvan: {model_prefix}.model")


class AndricDatasetSP(Dataset):
    def __init__(self, csv_path: str, spm_model_path: str = "data/spm_andric.model"):
        # Učitaj CSV
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
        self.pairs = list(zip(df["neutral_text"], df["original_text"]))

        # Učitaj SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, sentence: str) -> list[int]:
        # encode_as_ids automatski dodaje BOS i EOS
        return self.sp.encode(str(sentence), add_bos=True, add_eos=True)

    def decode(self, ids: list[int]) -> str:
        # Ukloni specijalne tokene pre dekodiranja
        clean = [i for i in ids if i not in (0, 1, 2, 3)]
        return self.sp.decode(clean)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        neutral, andric = self.pairs[idx]
        return self.encode(neutral), self.encode(andric)


# ── Brza provjera ──
if __name__ == "__main__":
    csv_path = "data/andric_style_transfer_dataset101.csv"

    # Treniraj SentencePiece ako model ne postoji
    if not os.path.exists("data/spm_andric.model"):
        print("Treniram SentencePiece model...")
        train_sentencepiece(csv_path, vocab_size=1000)

    dataset = AndricDatasetSP(csv_path)

    print(f"Veličina rečnika (SentencePiece): {dataset.vocab_size}")
    print(f"Broj parova: {len(dataset)}")

    # Primer tokenizacije
    test = "Pada kiša na grad."
    encoded = dataset.encode(test)
    decoded = dataset.decode(encoded)
    print(f"\nOriginalna rečenica: {test}")
    print(f"Tokeni (brojevi):    {encoded}")
    print(f"Tokeni (tekst):      {dataset.sp.id_to_piece(encoded)}")
    print(f"Dekodirana rečenica: {decoded}")

    # Poređenje sa word-level tokenizacijom
    print(f"\nPoređenje broja tokena:")
    print(f"  Word-level: {len(test.split())} tokena")
    print(f"  SentencePiece: {len(encoded) - 2} tokena (bez BOS/EOS)")