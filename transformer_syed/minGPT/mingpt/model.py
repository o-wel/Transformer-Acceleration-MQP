"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.attention_backend = getattr(config, "attention_backend", "eager")
        if self.attention_backend not in {"eager", "sdpa", "math", "flash"}:
            raise ValueError("attention_backend must be eager, sdpa, math, or flash")
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, past_key_value=None, use_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present = (k, v) if use_cache else None

        if self.attention_backend == "eager":
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, past_length:past_length + T, :past_length + T] == 0, float('-inf'))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v
        else:
            from contextlib import nullcontext
            from torch.nn.attention import SDPBackend, sdpa_kernel
            if self.attention_backend == "flash" and q.device.type != "cuda":
                raise RuntimeError("FlashAttention requires CUDA; use math or sdpa on CPU")
            context = nullcontext() if self.attention_backend == "sdpa" else sdpa_kernel(
                SDPBackend.FLASH_ATTENTION if self.attention_backend == "flash" else SDPBackend.MATH)
            # SDPA's is_causal is upper-left aligned. Cached queries belong at
            # the RIGHT end of K/V; one-token decode can see every cached key.
            mask = None
            causal = past_length == 0 and T > 1
            if past_length and T > 1:
                from torch.nn.attention.bias import causal_lower_right
                mask = causal_lower_right(T, k.size(2))
            with context:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, is_causal=causal,
                    dropout_p=self.attn_dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return (y, present) if use_cache else y

class LoRALinear(nn.Module):
    """Frozen linear projection plus a trainable low-rank weight update."""

    def __init__(self, linear, rank=4, alpha=4):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        self.linear = linear
        self.linear.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.A = nn.Linear(linear.in_features, rank, bias=False).to(linear.weight)
        self.B = nn.Linear(rank, linear.out_features, bias=False).to(linear.weight)
        # Random A and zero B preserve the pretrained output at initialization.
        nn.init.zeros_(self.B.weight)
        self.train(linear.training)

    def forward(self, x):
        return self.linear(x) + self.scale * self.B(self.A(x))

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))

    def mlpf(self, x):
        m = self.mlp
        return m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, past_key_value=None, use_cache=False):
        attention = self.attn(self.ln_1(x), past_key_value, use_cache)
        if use_cache:
            attention, present = attention
        x = x + attention
        x = x + self.mlpf(self.ln_2(x))
        return (x, present) if use_cache else x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.attention_backend = "eager"  # eager | math | sdpa (automatic) | flash (forced)
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type, *, lora_rank=None, lora_alpha=4):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # Causal masks are buffers, not learned weights. Some transformers versions
        # omit them from the checkpoint; minGPT creates its own in each attention layer.
        is_mask = lambda k: k.endswith(('.attn.bias', '.attn.masked_bias'))
        keys = [k for k in sd_hf if not is_mask(k)]
        model_keys = {k for k in sd if not is_mask(k)}
        assert set(keys) == model_keys, (set(keys) - model_keys, model_keys - set(keys))
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        if lora_rank is not None:
            model.enable_lora(rank=lora_rank, alpha=lora_alpha)
        return model

    def set_attention_backend(self, backend):
        """Switch kernels without changing weights, LoRA adapters, or cache format."""
        if backend not in {"eager", "sdpa", "math", "flash"}:
            raise ValueError("attention backend must be eager, sdpa, math, or flash")
        for block in self.transformer.h:
            block.attn.attention_backend = backend
        return self

    def enable_lora(self, rank=4, alpha=4):
        """Call after loading base weights and before constructing the optimizer.

        Adapt the fused Q/K/V, attention output, and both MLP projections.
        Only A/B weights train; embeddings, LayerNorms and the LM head freeze.
        To reload a LoRA state_dict, enable the same rank/alpha before loading it.
        """
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        if any(isinstance(m, LoRALinear) for m in self.modules()):
            raise ValueError("LoRA is already enabled")
        self.requires_grad_(False)
        for block in self.transformer.h:
            for parent, name in ((block.attn, 'c_attn'), (block.attn, 'c_proj'),
                                 (block.mlp, 'c_fc'), (block.mlp, 'c_proj')):
                setattr(parent, name, LoRALinear(getattr(parent, name), rank, alpha))
        return self

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, *, past_key_values=None, use_cache=False,
                last_token_only=False):
        """Forward tokens, optionally reusing an inference-only KV cache.

        The default return is the original (logits, loss). With use_cache=True,
        return (logits, None, cache), where cache is a tuple of (K, V) tensors,
        one per layer, each shaped (batch, heads, cached_tokens, head_width).
        Pass only NEW tokens alongside past_key_values. Caches are caller-owned,
        not checkpoint state: discard them after changing weights or prompts.
        Caching requires eval mode and no_grad/inference_mode, and accepts dense
        equal-length batches (no padding mask). Positions are offset by cache size.
        """
        device = idx.device
        b, t = idx.size()
        if t == 0:
            raise ValueError("At least one input token is required")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache and (self.training or torch.is_grad_enabled() or targets is not None):
            raise ValueError("KV caching requires eval(), no_grad/inference_mode, and no targets")
        if last_token_only and targets is not None:
            raise ValueError("last_token_only cannot be used with training targets")
        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != len(self.transformer.h):
                raise ValueError("Cache must contain one (K, V) pair per layer")
            for i, (pair, block) in enumerate(zip(past_key_values, self.transformer.h)):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("Each cache entry must be a (K, V) pair")
                k, v = pair
                if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                    raise ValueError("Cached K and V must be tensors")
                if k.ndim != 4 or v.shape != k.shape:
                    raise ValueError("Cached K and V must have matching four-dimensional shapes")
                if i == 0:
                    past_length = k.size(2)
                expected = (b, block.attn.n_head, past_length,
                            block.attn.n_embd // block.attn.n_head)
                if tuple(k.shape) != expected:
                    raise ValueError("Cache batch, heads, length, or head width does not match")
                if k.device != device or v.device != device or k.dtype != v.dtype:
                    raise ValueError("Cache device and K/V dtypes must match input device and each other")
        if past_length + t > self.block_size:
            raise ValueError("Input plus cache exceeds block_size; rebuild from the cropped context")
        pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device).unsqueeze(0)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        presents = []
        for i, block in enumerate(self.transformer.h):
            past = None if past_key_values is None else past_key_values[i]
            if use_cache:
                x, present = block(x, past_key_value=past, use_cache=True)
                presents.append(present)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        # Generation only needs the final position; apply this equally to both paths.
        logits = self.lm_head(x[:, -1:, :] if last_token_only else x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        if use_cache:
            return logits, loss, tuple(presents)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None,
                 use_cache=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        use_cache=True requires eval mode. At the context-window boundary, rebuild
        from the cropped window to preserve the original absolute-position behavior.
        """
        if use_cache and self.training:
            raise ValueError("Cached generation requires model.eval()")
        if idx.size(1) == 0:
            raise ValueError("Generation requires a nonempty prompt")
        past_key_values = None
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            if use_cache:
                if past_key_values is not None and past_key_values[0][0].size(2) < self.block_size:
                    logits, _, past_key_values = self(
                        idx[:, -1:], past_key_values=past_key_values,
                        use_cache=True, last_token_only=True)
                else:
                    # Sliding the window resets learned positions and changes every
                    # hidden state. Old K/V cannot be retained exactly in that case.
                    past_key_values = None
                    logits, _, past_key_values = self(
                        idx_cond, use_cache=True, last_token_only=True)
            else:
                logits, _ = self(idx_cond, last_token_only=True)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
