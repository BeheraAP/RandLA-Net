import torch
from torch import nn
import numpy

data = torch.tensor(numpy.loadtxt('./small.xyz', delimiter=' '),
	dtype = torch.float32)
locn = data[:,:3]
feat = data[:,3:]

class LocSE(nn.Module):
	def __init__(self, k, point_cloud):
		super().__init__()
		self.P = point_cloud
		self.K = k
		self.MLP = nn.Linear(10,6)


	def rel_pos_enc(self, i, knn_loc):
		pi  = self.P[i,:3]
		r = []
		for j in knn_loc:
			diff = pi-j
			dist = torch.norm(diff).unsqueeze(dim=0)
			cat = torch.cat((pi,j,diff,dist))
			r.append(self.MLP(cat))
		return torch.vstack(r)

	def knn(self, i):
		diff = self.P[:,:3]- self.P[i,:3]
		near_ks = diff.norm(dim=1).topk(self.K)
		near = torch.index_select(self.P, 0, near_ks.indices)
		return near

	def forward(self, i):
		N = self.knn(i)
		R = self.rel_pos_enc(i, N[:,:3])
		F = torch.cat((N,R), dim=1)
		return F



#debugging
print(LocSE(10, data).forward(42).size())

class AttentivePool(nn.Module):
	def __init__(self):
		super().__init__()
		pass

	def forward(self, x):
		pass

	def debug():
		pass

class DilRes(nn.Module):
	def __init__(self):
		super().__init__()
		pass

	def forward(self, x):
		pass

	def debug():
		pass

class LFA(nn.Module):
	def __init__(self):
		super().__init__()
		pass

	def forward(self, x):
		pass

	def debug():
		pass

class RandSample(nn.Module):
	def __init__(self):
		super().__init__()
		pass

	def forward(self, x):
		pass

	def debug():
		pass

