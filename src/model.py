import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RoPE ──────────────────────────────────────────────────────────────────────
class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) — rotira Q i K vektore umesto
    da dodaje pozicijski vektor na embedding. Bolja generalizacija na
    duže sekvence. Koriste ga LLaMA, Mistral i sl."""

    def __init__(self, d_k: int, max_len: int = 512):
        super().__init__()
        # Frekvencije za rotaciju — formula iz originalnog RoPE rada
        theta = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("theta", theta)
        self.max_len = max_len

    def _get_cos_sin(self, seq_len: int, device):
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, self.theta)
        return torch.cos(freqs), torch.sin(freqs)

    def _rotate(self, x, cos, sin):
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]
        rotated = torch.cat([x1 * cos - x2 * sin,
                             x1 * sin + x2 * cos], dim=-1)
        return rotated

    def forward(self, q, k):
        # q i k mogu imati različite dužine (cross-attention!)
        q_len = q.shape[2]
        k_len = k.shape[2]

        cos_q, sin_q = self._get_cos_sin(q_len, q.device)
        cos_k, sin_k = self._get_cos_sin(k_len, k.device)

        # Broadcast na (1, 1, seq_len, d_k/2)
        cos_q = cos_q.unsqueeze(0).unsqueeze(0)
        sin_q = sin_q.unsqueeze(0).unsqueeze(0)
        cos_k = cos_k.unsqueeze(0).unsqueeze(0)
        sin_k = sin_k.unsqueeze(0).unsqueeze(0)

        return self._rotate(q, cos_q, sin_q), self._rotate(k, cos_k, sin_k)


# ── GQA ───────────────────────────────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) — više Q glava deli iste K i V matrice
    po grupama. Manji broj parametara i brže generisanje. Koristi ga LLaMA 2/3.

    num_heads     = broj Q glava (npr. 4)
    num_kv_heads  = broj K/V glava (npr. 2) — mora deliti num_heads
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        # Q ima num_heads projekcija, K i V samo num_kv_heads
        self.W_q = nn.Linear(d_model, num_heads * self.d_k)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionalEncoding(self.d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch = Q.size(0)

        # Projekcija
        q = self.W_q(Q).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch, -1, self.num_kv_heads, self.d_k).transpose(1, 2)

        # RoPE na Q i K
        q, k = self.rope(q, k)

        # Proširi K i V na num_heads ponavljanjem po grupama
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        x = torch.matmul(attn_weights_dropped, v)

        # Spoji glave
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(x), attn_weights


# ── FeedForward ───────────────────────────────────────────────────────────────
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


# ── Encoder ───────────────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_output, _ = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_kv_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, num_kv_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ── Decoder ───────────────────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
        self.cross_attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attention(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout(attn_output)

        norm_x2 = self.norm2(x)
        attn_output, cross_attn_weights = self.cross_attention(
            norm_x2, encoder_output, encoder_output, src_mask
        )
        x = x + self.dropout(attn_output)

        ff_output = self.feed_forward(self.norm3(x))
        x = x + self.dropout(ff_output)
        return x, cross_attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_kv_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, num_kv_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        cross_attentions = []
        for layer in self.layers:
            x, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            cross_attentions.append(cross_attn)
        return self.norm(x), cross_attentions


# ── Transformer ───────────────────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 4,
                 num_kv_heads: int = 2, d_ff: int = 1024, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, num_kv_heads,
                               d_ff, num_layers, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, num_kv_heads,
                               d_ff, num_layers, dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src, pad_idx=0):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx=0):
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len,
                                            device=tgt.device)).bool()
        return pad_mask & causal_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        encoder_output = self.encoder(src, src_mask)
        decoder_output, cross_attentions = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        return self.output_projection(decoder_output), cross_attentions


# ── Provera ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vocab_size = 3531
    model = Transformer(vocab_size=vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Broj parametara: {total_params:,}")
    src = torch.randint(1, vocab_size, (2, 10))
    tgt = torch.randint(1, vocab_size, (2, 15))
    output, _ = model(src, tgt)
    print(f"Oblik izlaza: {output.shape}")
    print("Model radi ispravno!")