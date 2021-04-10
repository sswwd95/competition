import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print(torch.__version__)

# GeForce RTX 3090
# True
# 1.8.1+cu111