# Andrić Style Transfer

Projekat iz predmeta Inteligentni sistemi - stilski transfer teksta na srpskom
jeziku, transformacija neutralnih rečenica u književni stil Ive Andrića,
korišćenjem Encoder-Decoder Transformer arhitekture trenirane od nule na
ručno kreiranom datasetu (550 parova rečenica).

**Student:** Nikol Vidaković, RA232/2023, Fakultet tehničkih nauka, Novi Sad

Detaljna dokumentacija celog procesa, organizovana po fazama projekta, nalazi
se u [`documentation/Andric_Style_Transfer_Dokumentacija.pdf`](documentation/Andric_Style_Transfer_Dokumentacija.pdf).

## Struktura projekta

```
andric-style-transfer/
├── data/
│   └── andric_style_transfer_dataset101.csv  # dataset (550 parova)
├── src/
│   ├── model.py          # Transformer arhitektura (GQA + RoPE, od nule)
│   ├── dataset.py        # word-level tokenizer (baseline eksperimenti)
│   ├── dataset_mt5.py    # mT5 tokenizer sa kompaktnim remapiranjem
│   ├── train.py          # trening sa train/val splitom i early stopping
│   ├── inference.py      # generisanje teksta (Top-K sampling)
│   └── attention_viz.py  # vizualizacija cross-attention težina
├── data/
│   ├── loss_baseline.png        # grafici lossa po eksperimentima
│   ├── loss_experiment2.png
│   ├── loss_mt5_dropout.png
│   ├── loss_mt5_prefinal.png
│   ├── loss_mt5_final1.png
│   └── attention_*.png          # cross-attention heatmape
├── documentation/
│   └── Andric_Style_Transfer_Dokumentacija.pdf # detaljna dokumentacija
└── README.md
```

## Pokretanje

Projekat koristi `uv` za upravljanje paketima.

**Instalacija zavisnosti**
```
uv sync
```

**Trening modela** (učitava dataset, trenira model, čuva checkpoint sa
najboljim val lossom)
```
uv run src/train.py
```

**Generisanje teksta** (koristi već sačuvan model)
```
uv run src/inference.py
```

**Vizuelizacija attention mehanizma**
```
uv run src/attention_viz.py
```

## Pristup

Dataset od 550 parova podeljen je na trening (80%) i validacioni (20%) skup
uz fiksni seed (42). Tokenizacija koristi pre-trenirani mT5 tokenizer
(google/mt5-small) sa kompaktnim remapiranjem - od 250.100 originalnih tokena
uzima se samo 3531 koji se pojavljuju u datasetu, čime se embedding tabela
smanjuje sa ~64M na ~900K parametara.

Pet eksperimenata je sprovedeno redom, pri čemu je svaki motivisan nalazima
prethodnog: baseline (word-level tokenizer), produženi trening, uvođenje mT5
tokenizera, istraživanje hiperparametara (dropout, batch size, label
smoothing), i finalna arhitektura sa RoPE i GQA.

Early stopping prati isključivo validacioni loss (patience=30). Generisanje
je autoregresivno sa Top-K samplingom (k=50) i temperaturom T=0.8.

## Rezultati

**Finalni model:** Encoder-Decoder Transformer sa RoPE i GQA (~7.65M parametara)

| Konfiguracija | Val loss | Epohe |
|---|---|---|
| dropout=0.1 | 4.96 | 55 |
| dropout=0.3 (prefinal) | 4.75 | 91 |
| dropout=0.3 + label smoothing | 5.28 | 115 |
| dropout=0.3 + RoPE + GQA (finalni) | 4.82 | 60 |

Generisanje je nekoherentno na svim konfiguracijama: uzrok je veličina
dataseta (550 parova), a ne greška u arhitekturi. Vizuelizacija
cross-attention mehanizma potvrđuje da model ispravno čita ulaz i fokusira
pažnju na semantički ključne tokene.