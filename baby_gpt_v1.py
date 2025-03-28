# Script to iterate on reimplementing gpt starting from
# bigram language model. Start with defining hyperparams,
# go into building bloks and finish with training, evaluation,
# and generating a sample.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
block_size = 8
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
learining_rate = 1e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200
max_new_tokens = 500
n_embd = 32


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


# Attention
class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)
        aff = q @ k.transpose(-1, -2)  # (B, T, T)
        aff *= self.head_size**-0.5  # scale to preserve var
        # prevent communication with future tokens
        aff = torch.masked_fill(aff, self.tril[:T, :T] == 0, -torch.inf)
        aff = F.softmax(aff, dim=-1)
        # value
        v = self.value(x)  # (B, T, H)
        out = aff @ v  # (B, T, H)
        return out


# Multihead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(n_heads)])

    def forward(self, x):
        # x (B, T, C) and H = C / n_heads
        out = torch.concat([head(x) for head in self.heads], dim=-1)  # (B,T,C)
        return out


# Feed Forward
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(n_embd, n_embd * 4)
        self.w2 = nn.Linear(n_embd * 4, n_embd)
        self.act = nn.ReLU()

    def forward(self, x):
        # x (B,T,C)
        x = self.act(self.w1(x))  # (B,T,C*4)
        out = self.w2(x)  # (B,T,C)
        return out


# Block - multi-head attention followed by feed-forward
class Block(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, head_size)
        self.ff = FeedForward()

    def forward(self, x):
        # x (B,T,C)
        x = self.mha(x)  # (B,T,C)
        out = self.ff(x)  # (B,T,C)
        return out


# Model definition
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, n_embd)
        self.position_embed_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(4, n_embd // 4) for _ in range(3)]
        )  # 3 layers
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # see baby_gpt notebook for some exploration why default init
        # for linear layers isn't working that good.
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_embeds = self.token_embed_table(x)  # (B, T, C)
        pos_embeds = self.position_embed_table(torch.arange(T, device=device))  # (T, C)
        x = tok_embeds + pos_embeds  # (B, T, C)
        for block in self.blocks:
            x = block(x)  # (B, T, C)
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
model = BigramLanguageModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learining_rate)

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
gen_tokens = model.generate(
    torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens
)
print(decode(gen_tokens[0].tolist()))
