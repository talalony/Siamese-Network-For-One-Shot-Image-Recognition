import json
import argparse

import matplotlib.pyplot as plt

from model import SiameseVerifier
from training import train, test
from datasets import *

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device {device}")


def main():
    parser = argparse.ArgumentParser(description="Siamese Network Training and Testing")
    parser.add_argument('--train', type=str, help='Path to training data')
    parser.add_argument('--test', type=str, help='Path to testing data')

    args = parser.parse_args()

    model = SiameseVerifier()
    model.to(device)

    if args.train:
        train(args.train, model, device)
        torch.save(model.state_dict(), "model_omniglot_final.pth")

        with open("training_losses.json", "r") as f:
            losses = json.load(f)

        # Plot the losses
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    if args.test:
        model.load_state_dict(torch.load("model_omniglot_final.pth", map_location=device))
        test(args.test, model, device)


if __name__ == "__main__":
    main()
