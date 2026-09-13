import torch
import torch.nn as nn
import math

PAD_TOKEN = 0


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (can't look at future positions)."""

    def __init__(self, d_model, n_heads, max_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Precompute a causal mask: position i can only attend to positions <= i
        mask = torch.tril(torch.ones(max_len, max_len)).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, d_model

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape into heads: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)
        mask = self.causal_mask[:T, :T]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # back to (B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        # Residual connections around attention and feed-forward
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SequenceRecTransformer(nn.Module):
    """
    Tiny GPT-style decoder-only transformer, applied to item-ID sequences
    instead of text tokens.
    """

    def __init__(
        self,
        vocab_size,
        max_len=50,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # GPT-style small initialization — default nn.Embedding init (std=1.0)
        # produces huge, unstable dot-product scores at this dimensionality
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[PAD_TOKEN].fill_(0.0)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        # No separate output_head: we use token_emb.weight for the output
        # projection too (weight tying). This halves parameter count and is
        # what lets us score a SUBSET of items cheaply for negative sampling,
        # instead of always projecting to the full vocab.

    def get_hidden(self, x):
        """Returns final hidden states (B, T, d_model), before projecting to vocab."""
        B, T = x.shape

        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        tok_emb = self.token_emb(x)          # (B, T, d_model)
        pos_emb = self.pos_emb(positions)    # (1, T, d_model)

        h = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            h = block(h)

        h = self.ln_final(h)
        return h  # (B, T, d_model)

    def score_items(self, hidden, item_ids):
        """
        Score given hidden vectors against a specific set of item IDs.
        hidden: (N, d_model)
        item_ids: (K,) — shared across all N rows, OR (N, K) for per-row item sets
        Returns: (N, K) scores
        """
        item_emb = self.token_emb(item_ids)  # (K, d_model) or (N, K, d_model)
        scale = self.d_model ** 0.5
        if item_emb.dim() == 2:
            return (hidden @ item_emb.T) / scale  # (N, K)
        else:
            return torch.einsum("nd,nkd->nk", hidden, item_emb) / scale

    def score_full_vocab(self, hidden):
        """Score hidden vectors against the ENTIRE vocabulary. Use only for
        eval on small batches, or the final position, since this is the
        expensive full-softmax operation we're avoiding during training."""
        return (hidden @ self.token_emb.weight.T) / (self.d_model ** 0.5)  # (N, vocab_size)

    def forward(self, x):
        """Kept for compatibility: returns hidden states, NOT full logits.
        Use score_full_vocab() or score_items() explicitly on the output."""
        return self.get_hidden(x)