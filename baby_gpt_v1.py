# Script to iterate on reimplementing gpt starting from
# bigram language model. Start with defining hyperparams,
# go into building bloks and finish with training, evaluation,
# and generating a sample.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
block_size = 256
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
learining_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
max_new_tokens = 500
n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.2
# it stops learning without residual connections at n_layers == 6

# Load and prepare the data
with open("input.txt", mode="r") as f:
    words = f.read()

# Vocab and tokenizer
vocab = sorted(set(words))
vocab_size = len(vocab)
# mappings
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for i, c in enumerate(vocab)}
# encoder and decoder
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
# to tensor and train/test split
data = torch.tensor(encode(words), dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n]
test = data[n:]

torch.manual_seed(1337)  # for repro


# get batch of random examples
def get_batch(split):
    data = train if split == "train" else test
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# loss estimation helper
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros((eval_iters,))
        for i in range(eval_iters):
            Xb, Yb = get_batch(split)
            _, loss = model(Xb, Yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Multihead Attention (vectorized)
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.Nh = n_heads
        self.H = head_size
        self.attn = nn.Linear(n_embd, 3 * n_heads * head_size, bias=False)
        self.att_do = nn.Dropout(dropout)
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.res_do = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):
        B, T, C = x.shape  # H == C // Nh
        # project and split (B,T,C)@(C,Nh*H*3) -> (B,T,Nh*H*3) -> 3 (B,T,Nh*H)
        qs, ks, vs = self.attn(x).split(self.Nh * self.H, dim=2)
        # view as (B,T,Nh,H) and transpose to (B,Nh,T,H)
        qs = qs.view((B, T, self.Nh, self.H)).transpose(1, 2)
        ks = ks.view((B, T, self.Nh, self.H)).transpose(1, 2)
        vs = vs.view((B, T, self.Nh, self.H)).transpose(1, 2)
        # Token communication (B,Nh,T,H) @ (B,Nh,H,T) -> (B,Nh,T,T)
        affs = qs @ ks.transpose(2, 3)
        affs *= self.H**-0.5  # scale by sqrt(head size)
        # Prevent communication with future tokens
        affs = affs.masked_fill(self.tril[:T, :T] == 0, -torch.inf)
        affs = F.softmax(affs, dim=-1)  # token communicated affinities weights
        affs = self.att_do(affs)
        # weighted values
        # affs (B,Nh,T,T) @ vs (B,Nh,T,H) -> (B,Nh,T,H)
        y = affs @ vs
        # swap Nh and T and merge Nh dimension
        # (B,Nh,T,H) -> (B,T,Nh,H) -> (B,T,Nh*H)
        y = y.transpose(1, 2).reshape((B, T, self.Nh * self.H))
        # Project and residual dropout
        y = self.res_do(self.proj(y))
        return y


# Feed Forward
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x (B,T,C)
        out = self.net(x)
        return out


# Block - multi-head attention followed by feed-forward
class Block(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, head_size)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x (B,T,C)
        # post-norm
        x = self.ln1(x + self.mha(x))
        x = self.ln2(x + self.ff(x))
        return x


# Model definition
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, n_embd)
        self.position_embed_table = nn.Embedding(block_size, n_embd)
        self.emb_do = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_heads, n_embd // n_heads) for _ in range(n_layers)]
        )  # 3 layers
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # see baby_gpt notebook for some exploration why default init
        # for linear layers isn't working that good.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_embeds = self.token_embed_table(x)  # (B, T, C)
        pos_embeds = self.position_embed_table(torch.arange(T, device=device))  # (T, C)
        x = tok_embeds + pos_embeds  # (B, T, C)
        x = self.emb_do(x)
        for block in self.blocks:
            x = block(x)  # (B, T, C)
        x = self.ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            x = idx[:, -block_size:]  # feed only last block_size of inputs
            logits, loss = self(x)
            # only last time step
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=1)  # (B, C)
            next_i = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_i), dim=1)  # (B, T+1)
        return idx


# Initialize model and optimizer
model = GPTLanguageModel()
model.to(device)
# Split parameters to separately apply weight decay
# a bit hacky way wich rely on the model implementatio naming
gains_and_biases = []
others = []
for name, p in model.named_parameters():
    if "ln" in name or "bias" in name:
        gains_and_biases.append(p)
    else:
        others.append(p)
# set weight decay to 0 for all gains and biases
optimizer = torch.optim.AdamW(
    [{"params": others}, {"params": gains_and_biases, "weight_decay": 0.0}],
    lr=learining_rate,
    weight_decay=0.005,
)

# Train/eval loop
for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"step: {i}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}"
        )
    Xb, Yb = get_batch("train")  # get random batch of training examples
    logits, loss = model(Xb, Yb)  # do forward pass and compute loss
    optimizer.zero_grad(set_to_none=True)  # set grads of model parameters to None
    loss.backward()  # backward pass, compute grads
    optimizer.step()  # update model parameters

print("-------------")
# generate sample
model.eval()
gen_tokens = model.generate(
    torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens
)
print(decode(gen_tokens[0].tolist()))
