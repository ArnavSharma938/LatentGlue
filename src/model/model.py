import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

class SwiGLU(nn.Module):
    def forward(self, x):
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)

class ProteinProjection(nn.Module):
    def __init__(self, in_features=1152, out_features=768):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.proj = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(self.norm(x))

class LigandProjection(nn.Module):
    def __init__(self, in_features=768, out_features=768):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.proj = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(self.norm(x))

class SeedAttentionPooling(nn.Module):
    def __init__(self, dim=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, mask=None):
        query = self.query_norm(self.seed.expand(x.size(0), -1, -1))
        key_value = self.key_norm(x)
        key_padding_mask = None if mask is None else ~mask.bool()
        pooled, _ = self.attn(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            average_attn_weights=True,
        )
        return pooled[:, 0, :]

class RotaryJointBlock(nn.Module):
    def __init__(self, dim=384, n_heads=8, ffn_dim=1024, dropout=0.25):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")

        from esm.layers.rotary import RotaryEmbedding

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, ffn_dim * 2)
        self.ffn_act = SwiGLU()
        self.ffn_down = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, component_lengths, padding_mask=None):
        residual = x
        normalized = self.norm1(x)
        bsz, seq_len, _ = normalized.shape

        q = self.q_proj(normalized).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(normalized).view(bsz, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(normalized).view(bsz, seq_len, self.n_heads, self.head_dim)

        len_t, len_e, len_l = component_lengths
        q_t, q_e, q_l = torch.split(q, [len_t, len_e, len_l], dim=1)
        k_t, k_e, k_l = torch.split(k, [len_t, len_e, len_l], dim=1)

        q_t, k_t = self.rotary(q_t, k_t, seqlen_offset=0)
        q_e, k_e = self.rotary(q_e, k_e, seqlen_offset=10000)
        q_l, k_l = self.rotary(q_l, k_l, seqlen_offset=20000)

        q = torch.cat([q_t, q_e, q_l], dim=1)
        k = torch.cat([k_t, k_e, k_l], dim=1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=padding_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )

        attended = attended.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        x = residual + self.dropout(self.out_proj(attended))

        residual = x
        x = residual + self.dropout(self.ffn_down(self.ffn_act(self.ffn_up(self.norm2(x)))))
        return x

class RelationalPredictor(nn.Module):
    def __init__(
        self,
        in_dim=768,
        hidden_dim=384,
        n_layers=4,
        n_heads=8,
        ffn_dim=1024,
        dropout=0.25,
    ):
        super().__init__()
        self.mask_embeddings = nn.Embedding(3, in_dim)
        self.status_embed = nn.Embedding(2, hidden_dim)
        self.rotation_embed = nn.Embedding(3, hidden_dim)
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                RotaryJointBlock(
                    dim=hidden_dim,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, in_dim)

    def get_mask_embedding(self, component_id, batch_size, device, dtype):
        component_ids = torch.full((batch_size,), int(component_id), dtype=torch.long, device=device)
        return self.mask_embeddings(component_ids).to(dtype=dtype).unsqueeze(1)

    def _prepare_attn_mask(self, padding_mask):
        attn_mask = None
        if padding_mask is not None:
            if padding_mask.dim() == 2:
                attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = padding_mask
        return attn_mask

    def _add_conditioning(self, x, status_mask, rotation_index):
        x = self.proj_in(x)
        x = x + self.status_embed(status_mask.long())

        if rotation_index is None:
            rot_emb = self.rotation_embed.weight.mean(dim=0, keepdim=True).view(1, 1, -1)
        else:
            rotation_index = torch.as_tensor(rotation_index, device=x.device, dtype=torch.long)
            rot_emb = self.rotation_embed(rotation_index)
            if rot_emb.dim() == 1:
                rot_emb = rot_emb.view(1, 1, -1)
            else:
                rot_emb = rot_emb.unsqueeze(1)
        return x + rot_emb

    def _apply_padding_mask(self, x, padding_mask):
        if padding_mask is None:
            return x
        if padding_mask.dim() == 4:
            return x * padding_mask.squeeze(1).squeeze(1).unsqueeze(-1).to(dtype=x.dtype)
        return x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)

    def _encode_hidden(self, x, component_lengths, padding_mask=None):
        attn_mask = self._prepare_attn_mask(padding_mask)
        for layer in self.layers:
            x = layer(x, component_lengths, padding_mask=attn_mask)
        return self.norm_out(x)

    def forward(self, x, component_lengths, status_mask, rotation_index, padding_mask=None):
        x = self._add_conditioning(x, status_mask=status_mask, rotation_index=rotation_index)
        hidden = self._encode_hidden(x, component_lengths, padding_mask=padding_mask)
        return self._apply_padding_mask(self.proj_out(hidden), padding_mask)

class LatentGlueEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        assert torch.cuda.is_available(), "Training this model on CPU is not supported."
        self.device = torch.device(device)

        import esm.pretrained
        from transformers import AutoModel, AutoTokenizer

        backbone_gpu_dtype = torch.bfloat16

        self.esm_model = esm.pretrained.ESMC_600M_202412()
        if hasattr(self.esm_model, "transformer") and hasattr(self.esm_model.transformer, "_use_flash_attn"):
            self.esm_model.transformer._use_flash_attn = True
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.esm_model = self.esm_model.to(device=self.device, dtype=backbone_gpu_dtype)
        self.esm_model.eval()

        self.mol_tokenizer = AutoTokenizer.from_pretrained(
            "ibm-research/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
        )
        self.mol_model = AutoModel.from_pretrained(
            "ibm-research/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
            deterministic_eval=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        for param in self.mol_model.parameters():
            param.requires_grad = False
        self.mol_model = self.mol_model.to(device=self.device, dtype=backbone_gpu_dtype)
        self.mol_model.eval()

        self.role_embed = nn.Embedding(3, 768).to(self.device)
        self.prot_proj = ProteinProjection(1152, 768).to(self.device)
        self.mol_proj = LigandProjection(768, 768).to(self.device)
        self.target_pool = SeedAttentionPooling(dim=768, n_heads=8, dropout=0.1).to(self.device)
        self.effector_pool = SeedAttentionPooling(dim=768, n_heads=8, dropout=0.1).to(self.device)
        self.ligand_pool = SeedAttentionPooling(dim=768, n_heads=8, dropout=0.1).to(self.device)
        self._protein_backbone_cache = {}
        self._ligand_backbone_cache = {}

        self.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        self.esm_model.eval()
        self.mol_model.eval()
        return self

    def prepare_inputs(self, target_seqs, effector_seqs, smiles):
        with torch.no_grad():
            target_tokens = self.esm_model._tokenize(target_seqs).to(self.device)
            target_mask = target_tokens != self.esm_model.tokenizer.pad_token_id

            effector_tokens = self.esm_model._tokenize(effector_seqs).to(self.device)
            effector_mask = effector_tokens != self.esm_model.tokenizer.pad_token_id

            mol_inputs = self.mol_tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            mol_tokens = mol_inputs["input_ids"]
            mol_mask = mol_inputs["attention_mask"].bool()

        return target_tokens, target_mask, effector_tokens, effector_mask, mol_tokens, mol_mask

    def get_component_mask_token_id(self, component_id):
        if int(component_id) in (0, 1):
            token_id = getattr(self.esm_model.tokenizer, "mask_token_id", None)
        else:
            token_id = getattr(self.mol_tokenizer, "mask_token_id", None)
        if token_id is None:
            raise ValueError(f"Component {component_id} does not expose a mask token id.")
        return int(token_id)

    def _encode_protein_backbone(self, tokens, sequence_mask):
        x = self.esm_model.embed(tokens)
        for block in self.esm_model.transformer.blocks:
            x = block(x, sequence_mask, None, None, None)
            x = x * sequence_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x

    def _encode_ligand_backbone(self, tokens, padding_mask):
        x = self.mol_model.embeddings(input_ids=tokens)
        seq_len = tokens.size(1)
        extended_attention_mask = self.mol_model.get_extended_attention_mask(
            padding_mask.long(),
            (tokens.size(0), seq_len),
        )

        orig_dtype = x.dtype
        x = x.float()
        for block in self.mol_model.encoder.layer:
            x = block(x, attention_mask=extended_attention_mask)[0]
            x = x * padding_mask.unsqueeze(-1)
        return x.to(orig_dtype)

    def _get_backbone_cache(self, component_id):
        return self._protein_backbone_cache if int(component_id) in (0, 1) else self._ligand_backbone_cache

    def _encode_component_backbone(self, component_id, tokens, sequence_mask):
        component_id = int(component_id)
        if component_id in (0, 1):
            return self._encode_protein_backbone(tokens, sequence_mask)
        if component_id == 2:
            return self._encode_ligand_backbone(tokens, sequence_mask)
        raise ValueError(f"Unknown component_id={component_id}")

    def _get_cached_teacher_backbone_batch(self, component_id, keys, tokens, sequence_mask):
        if len(keys) != tokens.size(0):
            raise ValueError(
                f"Expected {tokens.size(0)} cache keys for component {component_id}, got {len(keys)}."
            )

        cache = self._get_backbone_cache(component_id)
        cached_rows = {}
        missing_groups = {}
        for idx, key in enumerate(keys):
            cached = cache.get(key)
            if cached is None:
                missing_groups.setdefault(key, []).append(idx)
            else:
                cached_rows[idx] = cached

        representative_indices = [indices[0] for indices in missing_groups.values()]
        fresh_backbones = None
        if representative_indices:
            fresh_backbones = self._encode_component_backbone(
                component_id,
                tokens[representative_indices],
                sequence_mask[representative_indices],
            ).detach()

        if fresh_backbones is not None:
            batch_backbones = fresh_backbones.new_zeros((tokens.size(0), tokens.size(1), fresh_backbones.size(-1)))
        elif cached_rows:
            sample = next(iter(cached_rows.values()))
            batch_backbones = torch.zeros(
                (tokens.size(0), tokens.size(1), sample.size(-1)),
                device=self.device,
                dtype=sample.dtype,
            )
        else:
            raise RuntimeError(f"Unable to build cached backbone batch for component {component_id}.")

        if fresh_backbones is not None:
            for row_offset, (key, indices) in enumerate(missing_groups.items()):
                rep_idx = indices[0]
                fresh_row = fresh_backbones[row_offset]
                batch_backbones[rep_idx] = fresh_row
                valid_len = int(sequence_mask[rep_idx].sum().item())
                cache[key] = fresh_row[:valid_len].detach().cpu()
                for dup_idx in indices[1:]:
                    batch_backbones[dup_idx, :valid_len] = fresh_row[:valid_len]

        for idx, cached_row in cached_rows.items():
            valid_len = cached_row.size(0)
            batch_backbones[idx, :valid_len] = cached_row.to(device=self.device, dtype=batch_backbones.dtype)

        return batch_backbones

    def forward_component(self, component_id, tokens, sequence_mask, cached_backbone=None):
        component_id = int(component_id)
        if component_id == 0:
            return self.forward_protein(tokens, sequence_mask, role_id=0, cached_backbone=cached_backbone)
        if component_id == 1:
            return self.forward_protein(tokens, sequence_mask, role_id=1, cached_backbone=cached_backbone)
        if component_id == 2:
            return self.forward_ligand(tokens, sequence_mask, role_id=2, cached_backbone=cached_backbone)
        raise ValueError(f"Unknown component_id={component_id}")

    def forward_protein(self, tokens, sequence_mask, role_id, cached_backbone=None):
        batch_size = tokens.size(0)
        if cached_backbone is not None:
            x = cached_backbone
        else:
            x = self._encode_protein_backbone(tokens, sequence_mask)

        backbone_latents = x.detach() if cached_backbone is None else None

        x_trunc = x[:, 1:, :]
        mask_trunc = sequence_mask[:, 1:].clone()
        eos_indices = mask_trunc.sum(dim=1) - 1
        mask_trunc[torch.arange(batch_size, device=x.device), eos_indices] = False

        projected = self.prot_proj(x_trunc)
        role = self.role_embed(
            torch.full((batch_size,), int(role_id), dtype=torch.long, device=x.device)
        ).unsqueeze(1)
        projected = projected + role
        return projected, mask_trunc, backbone_latents

    def forward_ligand(self, tokens, padding_mask, role_id=2, cached_backbone=None):
        batch_size = tokens.size(0)
        if cached_backbone is not None:
            x = cached_backbone
        else:
            x = self._encode_ligand_backbone(tokens, padding_mask)

        backbone_latents = x.detach() if cached_backbone is None else None

        x_trunc = x[:, 1:, :]
        mask_trunc = padding_mask[:, 1:].clone()
        eos_indices = mask_trunc.sum(dim=1) - 1
        mask_trunc[torch.arange(batch_size, device=x.device), eos_indices] = False

        projected = self.mol_proj(x_trunc)
        role = self.role_embed(
            torch.full((batch_size,), int(role_id), dtype=torch.long, device=x.device)
        ).unsqueeze(1)
        projected = projected + role
        return projected, mask_trunc, backbone_latents

    def pool_components(
        self,
        target_tokens,
        effector_tokens,
        ligand_tokens,
        target_mask,
        effector_mask,
        ligand_mask,
    ):
        return (
            self.target_pool(target_tokens, mask=target_mask),
            self.effector_pool(effector_tokens, mask=effector_mask),
            self.ligand_pool(ligand_tokens, mask=ligand_mask),
        )

    def forward(
        self,
        target_toks,
        effector_toks,
        mol_toks,
        target_mask,
        effector_mask,
        mol_mask,
        cached_backbones=None,
        compute_pools=True,
        cache_keys=None,
    ):
        if cached_backbones is None and cache_keys is not None:
            if len(cache_keys) != 3:
                raise ValueError(f"Expected 3 cache key groups, got {len(cache_keys)}.")
            cached_backbones = (
                self._get_cached_teacher_backbone_batch(0, cache_keys[0], target_toks, target_mask),
                self._get_cached_teacher_backbone_batch(1, cache_keys[1], effector_toks, effector_mask),
                self._get_cached_teacher_backbone_batch(2, cache_keys[2], mol_toks, mol_mask),
            )

        cached_target = None if cached_backbones is None else cached_backbones[0]
        cached_effector = None if cached_backbones is None else cached_backbones[1]
        cached_ligand = None if cached_backbones is None else cached_backbones[2]

        target_tokens, target_mask_tr, bb_t = self.forward_component(
            0,
            target_toks,
            target_mask,
            cached_backbone=cached_target,
        )
        effector_tokens, effector_mask_tr, bb_e = self.forward_component(
            1,
            effector_toks,
            effector_mask,
            cached_backbone=cached_effector,
        )
        ligand_tokens, ligand_mask_tr, bb_l = self.forward_component(
            2,
            mol_toks,
            mol_mask,
            cached_backbone=cached_ligand,
        )

        component_pools = None
        if compute_pools:
            component_pools = self.pool_components(
                target_tokens,
                effector_tokens,
                ligand_tokens,
                target_mask_tr,
                effector_mask_tr,
                ligand_mask_tr,
            )

        return (
            (target_tokens, effector_tokens, ligand_tokens),
            component_pools,
            (target_mask_tr, effector_mask_tr, ligand_mask_tr),
            (bb_t, bb_e, bb_l),
        )
