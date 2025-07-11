import torch 
import torch.nn as nn 
from torch.nn import functional as F
#hyperparameters 
batch_size = 32 
block_size = 8 
max_iters = 3000 
eval_interval = 300
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
eval_iters = 200 
torch.manual_seed(1337)

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#train/val split
data = torch.tensor(encode(text), dtype= torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size,(batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x,y = x.to(device), y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split)
      logits,loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out



#bigram model definition
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embeddingtable = nn.Embedding(vocab_size,vocab_size)

  def forward(self,idx,targets=None):
    logits = self.token_embeddingtable(idx) #(B,T,C)
    if targets == None:
      loss = None 
    else:
      B,T,C  = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)

    return logits,loss

  def generate(self,idx,max_new_tokens):
    #idx ---> (B,T)
    for _ in range(max_new_tokens):
      logits,loss = self(idx)
      logits = logits[:,-1,:] #(B,C)
      probs = F.softmax(logits,-1)  #(B,C)
      idx_next = torch.multinomial(probs,num_samples=1) #(B,1)
      idx = torch.cat((idx,idx_next), dim = 1)  #(B,T+1)
    return idx


model = BigramLanguageModel()
m = model.to(device)

#create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)


for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']: .4f}, val loss {losses['val']: .4f}")

  xb,yb = get_batch('train')
  logits,loss = m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(loss.item())


context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))