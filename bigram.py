import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size=32
block_size=8
max_iters=5000
eval_intervals = max_iters // 10
eval_iters = 200
lr=1e-3
device="cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

#load our text data
with open("tiny-shakespeare.txt","r",encoding="utf-8") as f:
    text=f.read()

# vocabulary
chars=sorted(list(set(text)))
vocab_size=len(chars)

#encode , decoder function and maping
stoi={ch:i for i,ch in enumerate(chars)} # creating map like stoi["a"]=10
itos={i:ch for i,ch in enumerate(chars)} # creating map like itos[10]="a"

encode=lambda seq: [stoi[ch] for ch in seq] # encoder lambda function to encode the input seq
decode=lambda seqList: "".join([itos[i] for i in seqList]) # decoder lambda function to decode the output seq

# encode out text into pytorch tensor
data=torch.tensor(encode(text),dtype=torch.long)

# train and val set
n=int(0.8*len(data))
train_data=data[:n]
val_data=data[n:]


# function to get a batch of data
def get_batch(split):

    """
        if len(data)=10
        block_size=7
        batch_size=4

        ix=[1,2,0,2] that means , 
        there would be batch_size number of starting indices and the range should be from 0 to (10-7)-1=> 0 - 2
        
        x=[[1:1+block_size],[2:2+block_size],[0:0+block_size],[2:2+block_size]]
        y=[[2:2+block_size],[3:3+block_size],[1:1+block_size],[3:3+block_size]]

    """

    data=train_data if split=="train" else val_data

    ix=torch.randint(len(data)-block_size, (batch_size,)) # 0 to (len(data)-block_size)-1 : batch_size int numbers

    x=torch.stack([data[i:i+block_size] for i in ix]) # x torch shape : (batch_size x block_size)
    y=torch.stack([data[i+1:i+block_size+1] for i in ix]) # y torch shape : (batch_size x block_size)

    x,y=x.to(device=device),y.to(device=device)

    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



# bigram language model
class BiGramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets = None):
        """
            here idx => batch_size x block_size
            so logits => for every token => we would look up the embedding table row for that token
            logits => batch_size x block_size x vocab_size

        """
        logits = self.token_embedding_table(idx)

        B,T,C=logits.shape

        """
        B = batch_size
        T = tensor size / block_size
        C = vocab_size

        For classification, PyTorch's F.cross_entropy() expects:

        input  = (number_of_predictions, number_of_classes)
        target = (number_of_predictions)
        """

        if targets is None:
            loss =None
        else:
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)


        return logits,loss

    def generate(self,idx,max_new_tokens):

        for _ in range(max_new_tokens):
            """
                here idx => batch_size(B) x block_size(T)
            """
            logits,loss=self(idx) # logits => ( batch_size , block_size , vocab_size )
            """
            we would generate based on the last token
            so , our logits will be model's vocabulary scores for the last input token
            like : ( batch_size , vocab_size )
            this indicates the last token row of the embedding

            """
            logits=logits[:,-1,:] # logits => ( batch_size , vocab_size )
            """
                logits.shape[0]=batch_size
                logits.shape[1]=vocab_size
                we wanna apply softmax on the vocab probs , that is dim=1
            """
            probs=F.softmax(logits,dim=1) 

            # Randomly select a token according to the probabilities for each batch
            # why randomly -> for diverse response every time
            idx_next=torch.multinomial(probs,num_samples=1) # idx_next => ( batch_size x 1 )

            idx=torch.cat((idx,idx_next),dim=1) # concate the next token to the input token idx 

        return idx

model=BiGramLanguageModel(vocab_size=vocab_size)
m=model.to(device=device)

#optimizer
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

for iter in range(max_iters):
    if iter%eval_intervals==0:
        losses=estimate_loss()
        print(f"step {iter} loss is {losses}, val loss is {losses['val']}")

    xb,yb=get_batch('train')

    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context=torch.zeros((1,1),dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))