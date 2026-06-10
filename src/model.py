import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # d_model mora biti deljiv sa num_heads
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimenzija po headu

        # Linearne projekcije za Q, K, V i izlaz
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Osnovna formula: softmax(QK^T / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Projekcija i podela na heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        x = self.scaled_dot_product_attention(Q, K, V, mask)

        # Spajanje heads nazad
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Matrica pozicijskih kodova
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN: normalizacija PRE attention/feedforward, stabilniji za male datasete

        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)

        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention: dekoder gleda svoju sopstvenu rečenicu
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        # Cross-attention: dekoder gleda enkoder (neutralnu rečenicu)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Korak 1: Self-attention (dekoder gleda šta je do sad generisao)
        norm_x = self.norm1(x)
        attn_output = self.self_attention(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout(attn_output)

        # Korak 2: Cross-attention (dekoder gleda enkoder — neutralnu rečenicu)
        norm_x2 = self.norm2(x)
        attn_output = self.cross_attention(norm_x2, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(attn_output)

        # Korak 3: FeedForward
        ff_output = self.feed_forward(self.norm3(x))
        x = x + self.dropout(ff_output)

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 4,
                 d_ff: int = 1024, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src, pad_idx=0):
        # Maska za PAD tokene u ulaznoj rečenici
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx=0):
        # Maska za PAD tokene
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        # Kauzalna maska — dekoder ne sme da vidi buduće reči
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        return pad_mask & causal_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)


# PROVERA:
if __name__ == "__main__":
    vocab_size = 2266
    model = Transformer(vocab_size=vocab_size)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Broj parametara: {total_params:,}")

    src = torch.randint(1, vocab_size, (2, 10))
    tgt = torch.randint(1, vocab_size, (2, 15))
    output = model(src, tgt)
    print(f"Oblik izlaza: {output.shape}")
    print("Model radi ispravno!")
