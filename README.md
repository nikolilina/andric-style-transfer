# Andrić Style Transfer

Encoder-Decoder Transformer za stilski transfer teksta — transformacija
neutralnih srpskih rečenica u književni stil Ive Andrića.

Projekat je realizovan od nule u PyTorch-u, bez korišćenja gotovih
pre-treniranih modela, kao deo kursa Inteligentni sistemi (FTN Novi Sad).

---

## Struktura projekta

andric-style-transfer/

├── src/

│   ├── model.py          # Transformer arhitektura (od nule)

│   ├── dataset.py        # Word-level tokenizer (baseline)

│   ├── dataset_mt5.py    # mT5 tokenizer sa kompaktnim remapiranjem

│   ├── train.py          # Trening sa train/val splitom i early stopping

│   ├── inference.py      # Generisanje teksta

│   └── attention_viz.py  # Vizualizacija cross-attention težina

├── data/

│   ├── andric_style_transfer_dataset101.csv  # Dataset (550 parova)

│   ├── best_model_mt5.pt                     # Finalni model

│   ├── loss_mt5_final.png                    # Kriva greške

│   └── attention_*.png                       # Attention heatmape

├── RESEARCH_LOG.md  # Istraživački dnevnik

└── README.md

---

## Model

Encoder-Decoder Transformer, ~7.27M parametara:

| Parametar   | Vrednost |
|-------------|----------|
| d_model     | 256      |
| num_heads   | 4        |
| d_ff        | 1024     |
| num_layers  | 3        |
| dropout     | 0.3      |

---

## Dataset

550 parova rečenica (neutralna → Andrićev stil):
- 350 autentičnih parova — citati iz Andrićevih dela sa ručno pisanim neutralnim verzijama
- 200 sintetičkih parova — AI-generisani parovi u Andrićevom stilu, manuelno revidirani

---

## Tokenizacija

Koristi se `google/mt5-small` tokenizer sa kompaktnim remapiranjem —
od 250,100 originalnih mT5 tokena, u datasetu se pojavljuje samo 3531,
pa se embedding tabela smanjuje sa ~64M na ~900K parametara.

---

## Pokretanje

```bash
# Instalacija
uv sync

# Trening
uv run src/train.py

# Generisanje
uv run src/inference.py

# Attention vizualizacija
uv run src/attention_viz.py
```

---

## Rezultati

Finalni model (val loss: 4.79, epoha 82) pokazuje očekivan overfitting
za dataset ove veličine. Cross-attention vizualizacija potvrđuje da model
ispravno čita ulaz i fokusira pažnju na semantički ključne tokene.

Detaljna analiza svih eksperimenata dostupna je u `RESEARCH_LOG.md`.