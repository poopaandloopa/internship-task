from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from models import *
from data import EthanolLevelDataset
import pandas as pd

TEST_DATA_PATH = "./data/EthanolLevel_TEST.tsv"


def inference(model, test_loader):
    model.eval()

    for batch_idx, (data, target) in enumerate(test_loader):
        test_output = model(data)
        predictions = test_output.argmax(dim=1)
        print(classification_report(y_true=target.numpy(), y_pred=predictions.numpy(), digits=3))
        ConfusionMatrixDisplay.from_predictions(y_true=target.numpy(), y_pred=predictions.numpy())
        plt.show()


def main():
    model = ConvModelResNet(n_classes=4)
    model.load_state_dict(torch.load("./weights/ConvModelResNet/epoch_127.pth"))

    test_dataframe = pd.read_csv(TEST_DATA_PATH, sep="\t", header=None)
    test_dataset = EthanolLevelDataset(dataframe=test_dataframe)
    test_dataloader = DataLoader(test_dataset, batch_size=500)

    inference(model, test_dataloader)


if __name__ == "__main__":
    main()
