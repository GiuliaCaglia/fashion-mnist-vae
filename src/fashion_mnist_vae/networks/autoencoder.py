from torch import nn, optim

from fashion_mnist_vae import networks


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mainline = nn.Sequential(networks.Encoder(), networks.Decoder())

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.criterion = nn.MSELoss()

    def forward(self, X):
        return self.mainline(X)

    def train(self, data_loader, epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in data_loader:
                self.optimizer.zero_grad()
                X_hat = self.forward(batch)
                loss = self.criterion(X_hat, batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss) / len(batch)
            losses.append(epoch_loss)
            print("Epoch: {}/{}; Loss: {}".format(epoch + 1, epochs, epoch_loss))

        return losses
