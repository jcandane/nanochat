"""nanochat/variant_cache.py

Q-K-V(+g) cache and generation loop for the HMAP operator family
(attn_variant="hmap": DMAP / AMAP / HMAP and the (b,a) interior).

WHY A DIFFERENT CACHE. Standard attention scores a new query against cached
KEYS only, so a K/V cache suffices. The variants score every past position as
BOTH key and query:

    logits_ij = 1/2<m_i,m_j> - b/2<n_i,n_j> + a/2(<q_i,k_j> - <k_i,q_j>) + (1-a)/2(g_i - g_j)

with m = (q+k)/sqrt2, n = (q-k)/sqrt2, and g_j the per-position node potential.
The flux term needs q_j for every cached j; the kinetic terms need both q_j and
k_j; the potential needs g_j. So we cache (q, k, v, g) per layer, all
post-RoPE / post-QK-norm / post-1.2x, exactly as the operator sees them.

Memory: 1.5x a K/V cache (three (B,Tmax,H,D) tensors per layer plus a small
(B,Tmax,H) potential). For d32 (32 layers, 16 heads, D=128) at Tmax=2048,
bf16, B=1: ~ 32 * 3 * 2048*16*128 * 2 bytes ~= 800 MB. Fine on an A10G.

Interface mirrors what nanochat.gpt.GPT.forward expects from a cache:
    get_pos(), advance(T), prev_embedding, n_layers
plus the variant-specific:
    write(layer, T0, q, k, v, g)          # store new positions T0..T0+T-1
    get_layer_qkvg(layer, Tk)             # views over positions 0..Tk-1

USAGE (Space / sampling scripts):
    from nanochat.variant_cache import variant_generate
    for tok in variant_generate(model, tokens, max_tokens=200, temperature=0.8, top_k=50):
        ...
The standard-attention arm should keep using nanochat.engine.Engine (FA3 KV cache).
"""

import torch
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE


class VariantKVCache:
    def __init__(self, batch_size, n_layers, n_head, head_dim, max_len, device,
                 dtype=COMPUTE_DTYPE, needs_g=True):
        self.n_layers = n_layers
        self.max_len = max_len
        self.pos = 0
        self.prev_embedding = None          # used by GPT.forward's smear logic
        shape = (n_layers, batch_size, max_len, n_head, head_dim)
        self.q = torch.empty(shape, device=device, dtype=dtype)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.g = (torch.empty((n_layers, batch_size, max_len, n_head), device=device, dtype=dtype)
                  if needs_g else None)

    # --- nanochat KVCache-compatible surface ---
    def get_pos(self):
        return self.pos

    def advance(self, T):
        self.pos += T
        assert self.pos <= self.max_len, f"VariantKVCache overflow: {self.pos} > {self.max_len}"

    def reset(self):
        self.pos = 0
        self.prev_embedding = None

    # --- variant-specific ---
    def write(self, layer, T0, q, k, v, g):
        """q,k,v: (B,T,H,D) post-RoPE/QK-norm/1.2x; g: (B,T,H) or None."""
        T = q.shape[1]
        assert T0 + T <= self.max_len, f"VariantKVCache overflow at layer {layer}"
        self.q[layer, :, T0:T0 + T] = q
        self.k[layer, :, T0:T0 + T] = k
        self.v[layer, :, T0:T0 + T] = v
        if g is not None:
            assert self.g is not None, "cache built with needs_g=False but g was provided"
            self.g[layer, :, T0:T0 + T] = g
        elif self.g is not None:
            raise RuntimeError(
                "cache has g storage but this layer supplied g=None; "
                "operator/cache alpha convention is inconsistent"
            )

    def get_layer_qkvg(self, layer, Tk):
        q = self.q[layer, :, :Tk]
        k = self.k[layer, :, :Tk]
        v = self.v[layer, :, :Tk]
        g = self.g[layer, :, :Tk] if self.g is not None else None
        return q, k, v, g


@torch.inference_mode()
def variant_generate(model, tokens, max_tokens, temperature=1.0, top_k=None, seed=42,
                     max_len=None):
    """Prefill + cached decode for attn_variant='hmap' models. Yields ints.

    tokens: list[int] prompt (batch size 1). Mirrors GPT.generate()'s sampling
    semantics (temperature, top_k, seeded multinomial) but runs in O(T) per
    step instead of re-forwarding the whole prefix.
    """
    assert isinstance(tokens, list)
    assert tokens, "variant_generate requires a non-empty prompt"
    cfg = model.config
    assert cfg.attn_variant == "hmap", "variant_generate is for the hmap family; use Engine for standard attention"
    device = model.get_device()
    max_len = max_len or cfg.sequence_len
    assert len(tokens) + max_tokens <= max_len, "prompt + max_tokens exceeds cache length"

    head_dim = cfg.n_embd // cfg.n_head
    cache = VariantKVCache(
        batch_size=1, n_layers=cfg.n_layer, n_head=cfg.n_head, head_dim=head_dim,
        max_len=max_len, device=device, dtype=COMPUTE_DTYPE, needs_g=(cfg.hmap_alpha < 1.0),
    )

    rng = None
    if temperature > 0:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

    def _sample(logits):
        logits = logits[:, -1, :]
        if top_k is not None and top_k > 0:
            vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < vals[:, [-1]], float("-inf"))
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1, generator=rng)
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Prefill: whole prompt in one forward (GPT.forward handles T>1 with a cache:
    # smear applied to positions 1+, rotary offset from cache.get_pos()).
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits = model.forward(ids, kv_cache=cache, logits_last_only=True)
    next_id = _sample(logits)

    for step in range(max_tokens):
        tok = next_id.item()
        yield tok
        if step + 1 == max_tokens:
            break
        # Decode: one token only. Rotary offset comes from cache.get_pos();
        # smear continuity comes from cache.prev_embedding.
        logits = model.forward(
            next_id,
            kv_cache=cache,
            logits_last_only=True,
        )
        next_id = _sample(logits)
