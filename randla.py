import torch
from torch import nn
from torch.utils.data import Dataset

import numpy
from lfa import LFA, RandSample

class PointCloudRR():
	def __init__(self, filename="./RRT2_group1_densified_point_cloud.xyz"):
		# self.data = pandas.read_table(
		# 	filename,
		# 	delimiter=' ',
		# 	names = ['X', 'Y', 'Z', 'R', 'G', 'B'],
		# )
		self.data = torch.tensor(numpy.loadtxt(filename, delimiter=' '))

	def size(self):
		return self.data.size()
	def __repr__(self):
		return '''Rajarani dataset:
    type:    point cloud
    format:  xyz
    size:    %d
    headers: X Y Z R G B'''%len(self.data)


class RandLA(nn.Module):
	def __init__(self, size=[1000,6]):
		super().__init__()
		N = size[0]
		d = size[0]
		self.layers = nn.ModuleList([
			nn.Linear(N, N),
		])
		pass

	def forward(self, x):
		print(x.size())
		pass

	def debug(self):
		pass

RandLA().debug()

