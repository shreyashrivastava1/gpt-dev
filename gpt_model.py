import torch 
import torch.nn as nn 
from torch.nn import functional as F

#hyperparameters 
batch_size = 64 
block_size = 256 
max_iters = 5000 
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200 
n_layer = 4
n_head = 6
n_embd = 384
dropout = 0.2
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#--------------
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

class Head(nn.Module):
  
  """one head of self attention"""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd,head_size, bias = False)
    self.query = nn.Linear(n_embd,head_size, bias = False)
    self.value = nn.Linear(n_embd,head_size, bias = False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    self.dropout = nn.Dropout(dropout)
  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x)    
    q = self.query(x) 

    wei = q @ k.transpose(-2,-1)/ C ** 0.5
    wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
    wei = F.softmax(wei, dim = 2)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):

    """multiple heads of self attention in parallel"""

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
       out = torch.cat([h(x) for h in self.heads],dim=-1)
       out = self.dropout(self.proj(out))
       return out


class FeedForward(nn.Module):
    def __init__(self,n_embd):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(n_embd,4 * n_embd),
         nn.ReLU(),
         nn.Linear(4 * n_embd,n_embd),
         nn.Dropout(dropout),
         )
       
    def forward(self,x):
      return self.net(x)


class Block(nn.Module):
   """Transformer Block: Communication followed by computation"""

   def __init__(self,n_embd,n_head):
      super().__init__()
      head_size = n_embd//n_head
      self.sa = MultiHeadAttention(n_head,head_size)
      self.ffwd = FeedForward(n_embd)
      self.ln1 = nn.LayerNorm(n_embd)
      self.ln2 = nn.LayerNorm(n_embd)
   
   def forward(self,x):
      x = x + self.sa(self.ln1(x))
      x = x + self.ffwd(self.ln2(x))            #here, B,T act as batch dimensions
      return x
   





#bigram model definition
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embeddingtable = nn.Embedding(vocab_size,n_embd)
    self.position_embeddingtable = nn.Embedding(block_size,n_embd)
    self.blocks = nn.Sequential(*
       [Block(n_embd,n_head=n_head) for _ in range(n_layer)]
    )
    self.ln_f =  nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd,vocab_size)

  def forward(self,idx,targets=None):
    #idx,targets are both (B,T) tensor of integers
    B,T = idx.shape
    tok_emb = self.token_embeddingtable(idx) #(B,T,C)
    pos_emb = self.position_embeddingtable(torch.arange(T,device=device)) # (T,C)
    x = tok_emb + pos_emb   #(B,T,C)
    x = self.blocks(x)  #(B,T,C)
    logits = self.lm_head(x) #(B,T,vocab_size)

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
      idx_cond = idx[:,-block_size:]    #crop idx to the last block_size tokens
      logits,loss = self(idx_cond)
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