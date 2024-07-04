import torch

#
#   Configuration class, contains all the important information for the project.
#
class Config:
    DATA_DIR = '../data/'
    CHECKPOINT_DIR = '../checkpoints/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    epochs = 24
    lr = 0.01

cfg = Config()