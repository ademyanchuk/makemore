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
learining_rate = 1e-2
max_iters = 3000
eval_interval = 300
eval_iters = 200
max_new_tokens = 500


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


# Model definition
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_lookup_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.embedding_lookup_table(x)  # (B, T, C)
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
            logits, loss = self(idx)
            # only last time step
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=1)  # (B, C)
            next_i = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_i), dim=1)  # (B, T+1)
        return idx


# Initialize model and optimizer
model = BigramLanguageModel(vocab_size)
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
