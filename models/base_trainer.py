import torch
import torch.nn as nn


class BaseTrainer(nn.Module):
    def __init__(self):
        super().__init__()

    def build_model(self):
        raise NotImplementedError

    def train_on_batch(self, X, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        raise NotImplementedError

    def update_nn_weights(self, optimizer, loss_manager, valid_mode=False):
        if not valid_mode:
            loss_manager.updateLossHistory()
        loss = loss_manager.computeTotalLoss()
        if not valid_mode:
            loss.backward()
            optimizer.step()
        else:
            pass
        loss = loss.item()
        return loss


if __name__ == "__main__":
    print("Start")
