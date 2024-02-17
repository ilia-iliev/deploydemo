from preprocessing import Tokenizer, TokenizingDataset
from datasets import load_dataset
from model import SentimentClassifier
import torch


def train():
     dataset = "sepidmnorozy/Bulgarian_sentiment"
     loaded_ds = load_dataset(dataset, split="train")

     vocab_size = 350    # number of tokens
     tokenizer = Tokenizer(vocab_size)
     tokenizer.fit(loaded_ds['text'])
     train_ds = TokenizingDataset(loaded_ds, tokenizer)     
     train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)

     model = SentimentClassifier(d_model = 8,
                                 vocab_size = vocab_size,
                                 n_heads=4,
                                 seq_len=85)
     criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.9, 0.1]))
     optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)

     for epoch in range(20):
          epoch_loss = 0
          epoch_correct = 0
          epoch_count = 0
          for batch in iter(train_dataloader):
               tokens,mask,label = batch
               predictions = model(tokens, mask)

               loss = criterion(predictions, label)

               correct = predictions.argmax(axis=1) == label
               acc = correct.sum().item() / correct.size(0)

               epoch_correct += correct.sum().item()
               epoch_count += correct.size(0)

               epoch_loss += loss.item()

               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
               optimizer.step()

          print(f"{epoch_loss=}")
          print(f"epoch accuracy: {epoch_correct / epoch_count}")
     
     torch.save(model, "model.pt")
     tokenizer.save("tokenizer.pkl")



if __name__ == "__main__":
    train()

