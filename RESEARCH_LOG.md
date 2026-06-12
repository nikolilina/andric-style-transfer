# Istraživački dnevnik — Stilski transfer teksta u duhu Ive Andrića

---

## 1. Uvod i cilj projekta

Cilj ovog projekta je istraživanje mogućnosti automatske transformacije
neutralnih, svakodnevnih rečenica na srpskom jeziku u rečenice koje stilski
podsećaju na prozu Ive Andrića — bogate epitetima, metaforama, filozofskim
tonom i karakteristično dugim, slojevitim rečenicama.

Model se trenira od nule, na sopstvenom, ručno kreiranom datasetu, bez
korišćenja gotovih, već istreniranih jezičkih modela. Fokus nije na
numerički savršenom rezultatu, već na razumevanju procesa: kako model uči,
gde greši, zašto greši, i kako se ponašanje menja sa promenom parametara i
količine podataka.

---

## 2. Arhitektura modela

Implementiran je **Encoder-Decoder Transformer od nule**, u PyTorch-u.
Svaka komponenta je posebna, samostalna celina u fajlu `src/model.py`, što
olakšava organizaciju, čitljivost i kasnije eventualne izmene. Komponente su:

- **MultiHeadAttention** — mehanizam pažnje sa 4 "glave" (`num_heads = 4`),
  implementiran ručno preko formule `softmax(QK^T / sqrt(d_k)) * V`. Svaka
  glava može da se fokusira na drugačiji tip veze između reči (npr. jedna na
  gramatičke odnose, druga na stilske obrasce).
- **PositionalEncoding** — sinusno/kosinusno kodiranje pozicije reči u
  rečenici, neophodno jer Transformer (za razliku od GRU/RNN modela) ne
  obrađuje rečenicu reč po reč, pa sam po sebi ne "zna" njihov redosled.
- **FeedForward** — dvoslojna mreža sa GELU aktivacijom (`d_ff = 1024`,
  4 puta veće od `d_model`), standardna praksa za moderne Transformer
  arhitekture.
- **EncoderLayer / DecoderLayer** — implementirani u **Pre-LN**
  konfiguraciji (normalizacija pre ulaska u attention/feedforward blok), što
  je stabilnije za treniranje na malim datasetima.
- **DecoderLayer** dodatno sadrži **cross-attention**, kojim dekoder "gleda"
  izlaz enkodera (originalnu neutralnu rečenicu) prilikom generisanja
  Andrićeve verzije.
- **Maskiranje** — `src_mask` ignoriše prazne (padding) tokene, a `tgt_mask`
  je takozvana kauzalna maska koja sprečava dekoder da "vidi" buduće tokove
  tokom treninga.

**Finalna konfiguracija modela:**

| Parametar     | Vrednost |
|---------------|----------|
| d_model       | 256      |
| num_heads     | 4        |
| d_ff          | 1024     |
| num_layers    | 3        |
| dropout       | 0.1      |
| Broj parametara | ~7.27M |

**Tokenizacija:** rečnik je izgrađen na nivou celih reči, direktno iz
dataseta, bez generičkog tokenizera (kakvi se koriste kod velikih modela tipa
GPT/BERT). Ovo je jednostavnije za implementaciju i razumevanje, ali ima
ograničenja — srpski jezik ima bogatu morfologiju (padeži, rodovi, glagolski
oblici), pa se mnoge reči u malom datasetu javljaju samo jednom ili dvaput, što
modelu teži učenje. Korišćenje generičkog tokenizera koji deli reči na manje,
ponovljive delove (tzv. subword tokenizacija) je realan pravac za poboljšanje
i razmatra se kao mogući budući eksperiment.

Specijalni tokeni: `[PAD]=0`, `[UNK]=1`, `[BOS]=2`, `[EOS]=3`.

**Trening:** funkcija greške `CrossEntropyLoss` (ignoriše PAD tokene),
optimizator **AdamW** (lr=0.0001, weight_decay=0.01), ograničavanje
gradijenta (gradient clipping, max_norm=1.0).

**Generisanje:** autoregresivno, token po token, sa **Top-K samplingom**
(k=50) i **temperaturom** (T=0.8), kako bi izlaz bio manje repetitivan i
manje rigidan u odnosu na uvek-najverovatniji izbor.

---

## 3. Dataset

- **Format:** CSV (`;` separator, UTF-8), kolone: `id`, `original_text`
  (Andrićeva rečenica), `neutral_text` (ručno napisana neutralna verzija).
- **Veličina:** dataset se postepeno skalira tokom projekta. Baseline
  verzija (Eksperiment 1) sadržala je **101 par rečenica**.

---

## Eksperiment 1 — Baseline trening (101 par, 100 epoha)

**Cilj eksperimenta:** Provjeriti da li ceo proces (dataset → model →
trening → generisanje) funkcioniše od početka do kraja, i dobiti prvu
referentnu vrednost funkcije greške (loss) za poređenje sa budućim
eksperimentima.

**Konfiguracija:** d_model=256, num_heads=4, d_ff=1024, num_layers=3,
dropout=0.1, batch_size=16, lr=0.0001, AdamW, 100 epoha, dataset: 101 par.

**Rezultati treninga (loss kroz epohe):**

| Epoha | Loss   |
|-------|--------|
| 10    | 6.5460 |
| 20    | 5.7871 |
| 30    | 4.9882 |
| 40    | 4.2545 |
| 50    | 3.5942 |
| 60    | 3.0134 |
| 70    | 2.4725 |
| 80    | 2.0086 |
| 90    | 1.5895 |
| 100   | 1.2341 |

Graf: `data/loss_baseline.png`

**Zapažanje o krivoj greške:** Loss opada postepeno i bez naglih skokova ili
zastoja tokom svih 100 epoha. Pad je gotovo linearan u poslednjih 60-70
epoha, što sugeriše da model u trenutku prekida treninga **još uči** i nije
dostigao stagnaciju — odlučujuća informacija za Eksperiment 2 (produženi
trening).

**Generisanje — primeri:**

| Ulaz (neutralna rečenica)     | Izlaz modela |
|--------------------------------|--------------|
| Pada kiša na grad.             | "samo se nijema nikad u jednu boju i nadmoć u trenutku ima čestitih to nije od vrednosti smo da se upotpunjuju kao spomenik da se i naš bolje svršiti." |
| Čovek hoda ulicom.              | "tvoji prihodi je sposoban se, ga za obrađivanje i ne se često, društva u kotlovima nemir; živo stvorenje. mu je stvoreno, nezdrave kad smo će već između palca" |
| Sunce zalazi iza brda.           | "svašta je brujanje svoje svoj ni najtvrđem tome, i istakoše žuto, izbrazdano kad snaga to se mnogo svega onoga i nezdrav" |

**Analiza izlaza:**

- **Pozitivno:**
  - Model generiše isključivo postojeće srpske reči iz rečnika — nema
    izmišljenih reči.
  - Vokabular je tematski i stilski u duhu Andrića: "nijema", "nadmoć",
    "stvorenje", "izbrazdano", "spomenik" — reči koje se ne bi pojavile u
    neutralnom registru.
  - Interpunkcija (zarezi, tačka-zapeta, tačke) se javlja na mestima koja
    liče na granice celina u rečenici — model je delimično uhvatio ritam
    rečenice.
  - Model nije generisao identičnu kopiju nijedne rečenice iz dataseta —
    nema čistog "papagajskog" ponavljanja u ovoj fazi.

- **Negativno:**
  - Rečenice su gramatički i smisaono nekoherentne — nema jasnog subjekta,
    predikata, slaganja po rodu/broju/padežu na nivou cele rečenice.
  - Izlaz je delimično nezavisan od ulaza — sve tri ulazne rečenice (kiša,
    čovek, sunce) generišu tematski sličan, apstraktan "filozofski šum" bez
    jasne veze sa konkretnim sadržajem ulazne rečenice. Mehanizam pažnje
    između enkodera i dekodera (cross-attention) vrlo verovatno još nije
    naučio smislenu vezu između ulaza i izlaza.
  - Rečenice su predugačke i "lutaju" — nema jasnog zaključka, podsećaju na
    tok svesti bez strukture.

**Pretpostavka o uzroku:**

Najverovatniji uzrok nekoherentnosti je kombinacija dva faktora:
1. Mali dataset (101 par) za zadatak ove kompleksnosti — model nema dovoljno
   primera da nauči **mapiranje** neutralno→Andrić, već uglavnom uči opštu
   "statistiku" Andrićevog vokabulara i interpunkcije.
2. Nedovršen trening — kriva greške još opada, model nije stabilizovan.

Trenutno ne možemo razlikovati koliki je doprinos svakog faktora — to je
predmet narednih eksperimenata.

**Sledeći koraci:**
1. **Eksperiment 2:** Trenirati identičan model duže (npr. 300-500 epoha) na
   istom datasetu od 101 para, da se vidi: (a) da li greška nastavlja da
   opada, stagnira, ili model počinje da "pamti" dataset napamet
   (prekomerno prilagođavanje); (b) da li generisanje postaje koherentnije
   ili samo "samouverenije" u istim greškama.
2. **Eksperiment 3:** Proširiti dataset na oko 200 parova i ponoviti baseline
   trening (100 epoha) radi poređenja sa Eksperimentom 1 — razdvojiti efekat
   veličine dataseta od efekta dužine treninga.

---

## Eksperiment 2 — (popuniti nakon treninga)

*(rezervisano za sledeći eksperiment)*
