import torch
from torch import nn
from torch.utils.data import Dataset
import pandas, numpy

class rajarani(Dataset):
	def __init__(self):
		super().__init__()
		data = pandas.read_table('./small.xyz', delimiter=' ')


	def __len__(self):
		pass

	def __getitem__(self):
		pass

	def __repr__(self):
		return "Rajarani dataset"

class RandLA(nn.Module):
	def __init__(self):
		super().__init__()
		pass

	def forward(self, x):
		pass

	def debug(self):
		pass

RandLA().debug()

