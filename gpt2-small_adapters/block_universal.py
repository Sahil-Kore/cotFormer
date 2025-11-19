import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Model
from datasets import load_dataset
import math
import random
import csv
import os
import time

# ---------- Config ----------
BACKBONE = "gpt2"
SEQ_LEN = 128
BATCH_SIZE = 4
NREPEAT = 3  # number of times the same block is applied (weight tying)
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
MAX_SAMPLES = 20000
SEED = 42
GRAD_CLIP = 1.0

torch.manual_seed(SEED)
random.seed(SEED)


# ---------- BlockUniversal layer (single weight-tied block) ----------
class BlockUniversalLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def build_causal_mask(self, S, device):
        mask = torch.full((S, S), float("-inf"), device=device)
        mask = torch.triu(mask, 1)  # block upper triangle (future) as -inf
        return mask

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, S, D) - both query and context are the same-length sequence (block-universal)
        key_padding_mask: (B, S) boolean, True where to mask
        """
        q = self.ln1(x)
        B, S, D = q.shape
        attn_mask = self.build_causal_mask(S, q.device)  # (S, S)
        attn_out, _ = self.mha(
            query=q,
            key=q,
            value=q,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


# ---------- Model wrapper (Block Universal Adapter on top of frozen backbone) ----------
class BlockUniversalAdapter(nn.Module):
    def __init__(self, backbone_name="distilgpt2", nrepeat=3, freeze_backbone=True):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(backbone_name)
        self.config = self.backbone.config
        self.embed_dim = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        num_heads = getattr(self.config, "n_head", 8)
        self.block = BlockUniversalLayer(
            self.embed_dim, num_heads=num_heads, ff_dim=self.embed_dim * 4, dropout=0.1
        )

        self.nrepeat = nrepeat
        self.post_ln = nn.LayerNorm(self.embed_dim)

        emb_w = self.backbone.get_input_embeddings().weight.detach().clone()
        self.lm_head = nn.Linear(self.embed_dim, emb_w.size(0), bias=False)
        self.lm_head.weight = nn.Parameter(emb_w)  # trainable copy

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, input_ids, attention_mask=None):
        # attention_mask: (B, S) with 1 for token, 0 for pad
        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids, dtype=torch.long, device=input_ids.device
            )

        with torch.no_grad():
            out = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            h = out.last_hidden_state.detach()  # (B, S, D)

        key_pad_bool = attention_mask == 0  # True where padding

        # Block Universal: repeatedly apply the same block to the previous representation
        prev = h
        for r in range(self.nrepeat):
            refined = self.block(
                prev, key_padding_mask=key_pad_bool
            )  # each repeat sees only prev (no concat)
            prev = self.post_ln(refined)

        final_repr = prev  # (B, S, D)
        logits = self.lm_head(final_repr)  # (B, S, V)
        return logits


# ---------- Tokenizer / Dataset ----------
def prepare_tokenizer(backbone_name=BACKBONE):
    tok = AutoTokenizer.from_pretrained(backbone_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_dataset(tokenizer, seq_len=SEQ_LEN, max_samples=MAX_SAMPLES):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.select(range(min(len(ds), max_samples)))
    texts = [t for t in ds["text"] if t and not t.isspace()]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    data = []
    for i in range(0, ids.size(0), BATCH_SIZE):
        data.append(
            {
                "input_ids": ids[i : i + BATCH_SIZE],
                "attention_mask": mask[i : i + BATCH_SIZE],
            }
        )
    return data


# ---------- Perplexity + time ----------
def compute_perplexity_and_time(model, tokenizer, dataset):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_t = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataset:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            t0 = time.time()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            total_t += time.time() - t0
            total_samples += input_ids.size(0)

            logits = logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous().clone()
            labels[labels == tokenizer.pad_token_id] = -100

            V = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="sum"
            )
            total_nll += loss.item()
            total_tokens += (labels.view(-1) != -100).sum().item()

    if total_tokens == 0:
        return float("inf"), 0.0
    ppl = math.exp(total_nll / total_tokens)
    avg_time = total_t / total_samples
    return ppl, avg_time


# ---------- Train + CSV logging ----------
def train_and_log():
    tokenizer = prepare_tokenizer(BACKBONE)
    data = build_dataset(tokenizer, seq_len=SEQ_LEN, max_samples=MAX_SAMPLES)
    val_split = int(0.9 * len(data))
    train_data = data[:val_split]
    val_data = data[val_split:]

    model = BlockUniversalAdapter(backbone_name=BACKBONE, nrepeat=NREPEAT)
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-2)

    print(f"Total Params: {total_params:,} | Trainable: {trainable_params:,}")

    start_time = time.time()
    model.train()
    step = 0

    for epoch in range(EPOCHS):
        random.shuffle(train_data)
        total_loss = 0.0
        for batch in train_data:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
            labels_flat = input_ids[:, 1:].reshape(-1)
            labels_flat[labels_flat == tokenizer.pad_token_id] = -100

            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += float(loss.item())
            step += 1
            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Avg Loss {total_loss / 50:.4f}")
                total_loss = 0.0

        val_ppl, _ = compute_perplexity_and_time(model, tokenizer, val_data)
        print(f"Epoch {epoch} | Validation PPL: {val_ppl:.2f}")

    training_time = time.time() - start_time
    val_ppl, avg_inf_time = compute_perplexity_and_time(model, tokenizer, val_data)
    avg_inf_ms = avg_inf_time * 1000.0

    # CSV write
    results_file = "gpt2_results.csv"
    header = [
        "model_name",
        "total_params",
        "trainable_params",
        "val_ppl",
        "training_time_s",
        "inference_time_ms",
    ]
    row = [
        "BlockUniversalAdapter",
        total_params,
        trainable_params,
        round(val_ppl, 2),
        round(training_time, 2),
        round(avg_inf_ms, 3),
    ]
    file_exists = os.path.exists(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    print(f"\nSaved metrics -> {results_file}")

    return model, tokenizer, val_data


# ---------- Greedy generation ----------
def generate_greedy(model, tokenizer, prompt, max_new_tokens=20):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated = input_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated)  # (1, S, V)
            next_logits = logits[:, -1, :]
            next_id = next_logits.argmax(dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


# ---------- Run ----------
if __name__ == "__main__":
    model, tokenizer, val_data = train_and_log()
    prompt = "In 2025 the field of machine learning"
    print(
        "Sample generation:",
        generate_greedy(model, tokenizer, prompt, max_new_tokens=30),
    )
