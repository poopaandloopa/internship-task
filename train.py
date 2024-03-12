import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import EthanolLevelDataset
from models import *
import os


TRAIN_DATA_PATH = "./data/EthanolLevel_TRAIN.tsv"
TEST_DATA_PATH = "./data/EthanolLevel_TEST.tsv"


def train(model, criterion, optimizer, train_loader, val_loader, test_loader, n_epochs):
    writer = SummaryWriter()
    if not os.path.exists(f"weights/{model.__class__.__name__}"):
        os.makedirs(f"weights/{model.__class__.__name__}")

    for epoch in range(n_epochs):
        print(f"Epoch #{epoch}")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            #data = data.reshape(data.shape[0], 1, data.shape[1])

            train_output = model(data)
            train_loss = criterion(train_output, target)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_accuracy = torch.sum(train_output.argmax(dim=1).eq(target)) / len(target)

            writer.add_scalar('Train loss',
                              scalar_value=train_loss.item(),
                              global_step=epoch*len(train_loader) + batch_idx)
            writer.add_scalar('Train accuracy on batch',
                              scalar_value=train_accuracy.item(),
                              global_step=epoch*len(train_loader) + batch_idx)

        model.eval()
        for batch_idx, (data, target) in enumerate(val_loader):
            #data = data.reshape(data.shape[0], 1, data.shape[1])

            val_output = model(data)
            val_loss = criterion(val_output, target)

            val_accuracy = torch.sum(val_output.argmax(dim=1).eq(target)) / len(target)

            writer.add_scalar('Val loss',
                              scalar_value=val_loss.item(),
                              global_step=epoch)
            writer.add_scalar('Val accuracy',
                              scalar_value=val_accuracy.item(),
                              global_step=epoch)

        torch.save(model.state_dict(), f=f'weights/{model.__class__.__name__}/epoch_{epoch}.pth')

    writer.close()


def main():
    train_dataframe = pd.read_csv(TRAIN_DATA_PATH, sep="\t", header=None)
    train_dataframe, val_dataframe = train_test_split(train_dataframe, test_size=0.2, random_state=42)
    test_dataframe = pd.read_csv(TEST_DATA_PATH, sep="\t", header=None)

    conv_model = ConvModel(n_classes=4)
    conv_resnet_model = ConvModelResNet(n_classes=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(conv_resnet_model.parameters(), lr=2e-4)

    train_dataset = EthanolLevelDataset(dataframe=train_dataframe)
    val_dataset = EthanolLevelDataset(dataframe=val_dataframe)
    test_dataset = EthanolLevelDataset(dataframe=test_dataframe)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=500)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    train(conv_resnet_model, criterion, optimizer, train_loader, val_loader, test_loader, 150)


if __name__ == "__main__":
    main()
