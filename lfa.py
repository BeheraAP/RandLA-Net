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
		self.MLP = nn.Linear(10,3)

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
		F = torch.cat((N[:,:3],R), dim=1)
		return F

# SharedMLP = nn.Sequential(
# 	nn.Linear(1,),
# 	nn.ReLU(),
# 	nn.Linear(),
# );

class AttentivePool(nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.attention_scores = torch.randn(shape, requires_grad=True)

	def forward(self, x):
		print("Input shape:", x.size())
		print("Attention score:", self.attention_scores.size())
		y = x*self.attention_scores
		print(y.size())
		print(y.sum(dim=1).size())
		print("Hadamard product: ",y.size())

#debugging
lse_out = LocSE(10, data).forward(42)
AttentivePool((10,6)).forward(lse_out)

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

