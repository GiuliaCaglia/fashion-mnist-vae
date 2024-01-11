import torch
from tqdm import tqdm


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mainline = torch.nn.Sequential(Encoder(), Decoder())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-2)
        self.criterion = torch.nn.MSELoss()

    def forward(self, X):
        return self.mainline(X)

    def train(self, data_loader, epochs):
        losses = []
        for epoch in tqdm(range(epochs), total=epochs):
            epoch_loss = 0
            for batch in data_loader:
                self.optimizer.zero_grad()
                X_hat = self.forward(batch)
                loss = self.criterion(X_hat, batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss) / len(batch)
            losses.append(epoch_loss)

        return losses
