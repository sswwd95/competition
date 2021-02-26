import torch
torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")