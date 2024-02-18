from preprocessing import Tokenizer, TokenizingDataset
from datasets import load_dataset
from model import SentimentClassifier
import torch


VOCAB_SIZE=350
SEQ_LEN=85
N_HEADS=4
D_MODEL=8
LR=1e-4
EPOCHS=20


def fit_model(ds):
     train_dataloader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
     model = SentimentClassifier(d_model = D_MODEL,
                                 vocab_size = VOCAB_SIZE,
                                 n_heads=N_HEADS,
                                 seq_len=SEQ_LEN)
     
     criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.9, 0.1]))     # more weight for 0 classes
     optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LR)
     for epoch in range(EPOCHS):
          epoch_loss = 0
          epoch_correct = 0
          epoch_count = 0
          for batch in iter(train_dataloader):
               tokens,mask,label = batch
               predictions = model(tokens, mask)

               loss = criterion(predictions, label)

               correct = predictions.argmax(axis=1) == label

               epoch_correct += correct.sum().item()
               epoch_count += correct.size(0)

               epoch_loss += loss.item()

               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
               optimizer.step()

          print(f"{epoch_loss=}")
          print(f"epoch accuracy: {epoch_correct / epoch_count}")
     return model

def train():
     dataset = "sepidmnorozy/Bulgarian_sentiment"
     loaded_ds = load_dataset(dataset, split="train")

     tokenizer = Tokenizer(VOCAB_SIZE)
     tokenizer.fit(loaded_ds['text'])
     train_ds = TokenizingDataset(loaded_ds, tokenizer)

     model = fit_model(train_ds)
     
     torch.save(model, "model.pt")
     tokenizer.save("tokenizer.pkl")

if __name__ == "__main__":
    train()

