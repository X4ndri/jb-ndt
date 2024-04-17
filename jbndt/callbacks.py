import torch


class EarlyStoppingAccuracy(torch.nn.Module):
    def __init__(self, patience=20):
        super().__init__()
        self.patience = patience
        self.counter = 0
        self.best_accuracy = None
        self.early_stop = False

    def forward(self, current_accuracy):
        if self.best_accuracy is None:
            self.best_accuracy = current_accuracy
        elif current_accuracy <= self.best_accuracy:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_accuracy = current_accuracy
            self.counter = 0
        return self.early_stop

