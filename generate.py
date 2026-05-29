import torch

from model import BigramLanguageModel
from data import decode
from config import *

model = BigramLanguageModel().to(device)

model.load_state_dict(torch.load('checkpoints/model.pt'))

model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device) # start with a single zero token
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))