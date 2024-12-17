"""
Full definition of a GPT Language Model, all of it in this single file with support for:-
- original architecture (uniform layer dimensions)
- diamond-shaped architecture (varying layer dimensions)
- unetxformer architecture (diamond-shaped + skip connections, inspired by U-Net)

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """ A single causal self-attention head. """

    def __init__(self, config, layer_dim, n_head):
        super().__init__()
        assert layer_dim % n_head == 0
        self.n_head = n_head
        self.layer_dim = layer_dim
        head_dim = layer_dim // n_head
        self.head_dim = head_dim

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(layer_dim, 3 * layer_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(layer_dim, layer_dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # compute query, key, values
        q, k, v = self.c_attn(x).split(self.layer_dim, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # causal attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """ A simple MLP block. """

    def __init__(self, config, layer_dim):
        super().__init__()
        self.c_fc = nn.Linear(layer_dim, 4 * layer_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * layer_dim, layer_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ A single Transformer block that can also handle dimension expansion/shrinking. """

    def __init__(self, config, prev_layer_dim, layer_dim, next_layer_dim, n_head):
        super().__init__()
        self.ln_1 = LayerNorm(layer_dim, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_dim, n_head)
        self.ln_2 = LayerNorm(layer_dim, bias=config.bias)
        self.mlp = MLP(config, layer_dim)

        self.prev_layer_dim = prev_layer_dim
        self.layer_dim = layer_dim
        self.next_layer_dim = next_layer_dim

        # Dimension expansion if layer_dim > prev_layer_dim
        if layer_dim > prev_layer_dim:
            # Parameter used to expand the dimension by concatenation
            self.expand_dim = nn.Parameter(torch.randn(layer_dim - prev_layer_dim))
        else:
            self.expand_dim = None

        # Dimension shrinking if next_layer_dim < layer_dim
        if next_layer_dim < layer_dim:
            self.shrink_dim = nn.Linear(layer_dim, next_layer_dim, bias=config.bias)
        else:
            self.shrink_dim = None

    def forward(self, x):
        # Expand dimension if needed
        if self.expand_dim is not None:
            extra_dim = self.expand_dim.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)
            x = torch.cat([x, extra_dim], dim=-1)

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        # Shrink dimension if needed
        if self.shrink_dim is not None:
            x = self.shrink_dim(x)

        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    layer_dims: list = None
    n_heads: list = None
    dropout: float = 0.0
    bias: bool = True
    model_architecture: str = 'original'  # 'original', 'diamond', 'unetxformer'
    use_unet: bool = False

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.layer_dims[0]),
            wpe = nn.Embedding(config.block_size, config.layer_dims[0]),
            drop = nn.Dropout(config.dropout),
        ))

        # Construct the transformer blocks
        blocks = []
        for i in range(len(config.layer_dims)):
            prev_dim = config.layer_dims[i-1] if i > 0 else config.layer_dims[0]
            curr_dim = config.layer_dims[i]
            next_dim = config.layer_dims[i+1] if i < len(config.layer_dims)-1 else config.layer_dims[-1]
            blocks.append(Block(config, prev_dim, curr_dim, next_dim, config.n_heads[i]))

        self.transformer.h = nn.ModuleList(blocks)
        self.transformer.ln_f = LayerNorm(config.layer_dims[-1], bias=config.bias)
        self.lm_head = nn.Linear(config.layer_dims[-1], config.vocab_size, bias=False)

        # Setup for unetxformer if use_unet = True
        self.use_unet = config.use_unet
        if self.use_unet:
            # We'll consider the midpoint for skip connections
            self.mid_index = config.n_layer // 2
            # Prepare skip projections for concatenating encoder and decoder outputs
            self.skip_projections = nn.ModuleList()
            for i in range(self.mid_index):
                enc_dim = config.layer_dims[i]
                dec_dim = config.layer_dims[-(i+1)]
                # after concatenation (enc_dim + dec_dim), project back to dec_dim
                self.skip_projections.append(nn.Linear(enc_dim + dec_dim, dec_dim, bias=config.bias))

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * len(self.transformer.h)))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        if not self.use_unet:
            # Just run all blocks straight through
            for block in self.transformer.h:
                x = block(x)
        else:
            # Unetxformer: encoder + decoder with skip connections
            encoder_outputs = []
            # Encoder half
            for i in range(self.mid_index):
                x = self.transformer.h[i](x)
                encoder_outputs.append(x)

            # If odd number of layers, process the middle block
            if self.config.n_layer % 2 != 0:
                x = self.transformer.h[self.mid_index](x)
                decoder_start = self.mid_index + 1
            else:
                decoder_start = self.mid_index

            # Decoder half with skip connections
            for j in range(decoder_start, self.config.n_layer):
                enc_layer_idx = self.mid_index - 1 - (j - decoder_start)
                skip_x = encoder_outputs[enc_layer_idx]

                # fuse skip connection by concatenation + projection
                x = torch.cat([x, skip_x], dim=-1)
                x = self.skip_projections[self.mid_index - 1 - (j - decoder_start)](x)

                x = self.transformer.h[j](x)

        x = self.transformer.ln_f(x)

        # Language modeling head
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # at inference, only take last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if needed
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # For simplicity, if from_pretrained is used, we assume original architecture
        # You can modify this if needed.
        config_args['model_architecture'] = 'original'
        config_args['use_unet'] = False
        # layer_dims and n_heads inferred from n_embd/n_head
        n_layer = config_args['n_layer']
        n_head = config_args['n_head']
        n_embd = config_args['n_embd']
        layer_dims = [n_embd]*n_layer
        n_heads = [n_head]*n_layer
        config_args['layer_dims'] = layer_dims
        config_args['n_heads'] = n_heads

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), "Mismatched state dict keys"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # A heuristic estimate of model FLOPs utilization
        N = self.get_num_params()
        cfg = self.config
        L = len(cfg.layer_dims)
        T = cfg.block_size
        # Rough estimate of flops per token for attention+MLP
        flops_per_token = 6*N
        # This is a very rough approximation
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12 # A100 bfloat16 peak flops ~312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # Simple generation
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
