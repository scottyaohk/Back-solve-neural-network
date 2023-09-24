from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256), 
            nn.LeakyReLU(0.01),
            nn.Linear(256, 10)
            )
        self.sm = nn.Softmax(-1)
    
    def forward(self, X):
        X = X.view(X.size()[0], -1)
        return self.sm(self.layers(X))

    
def train():
    # hyperparameters
    BATCH_SIZE = 64
    num_epochs = 5
    lr = 0.01
    model_path = "./mnist_fnn.pt"
    # load mnist
    train_dataset = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset =  datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 
    model = FNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    #
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for X, Y in train_dataloader:
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            right = 0
            total = 0
            for X, Y in test_dataloader:
                pred = model(X)
                r = torch.argmax(pred, -1) == Y
                right += torch.sum(r)
                total += len(r)
            acc = right/total
        print(f"Accuracy: {acc.item()*100:.2f}%")
        torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
    train()