import torch
from torch import nn
from torch import linalg

print("CUDA Available: ", torch.cuda.is_available())
class LocSE(nn.Module):
	def __init__(self, k, d):
		super().__init__()
		self.k = k
		self.d = d
		self.MLP = nn.Linear(in_features= 10, out_features = d)

	def gather_neighbor(self, xyz_feat, p_idx):
		assert self.k<=len(xyz_feat),\
		"k = %d should be less than n = %d"%(self.k, len(xyz_feat))

		xyz = xyz_feat[:,:3]
		p = xyz_feat[p_idx,:3]
		diff = xyz-p
		norm = linalg.norm(xyz-p, dim=1)
		_, neighbor = norm.sort()
		return neighbor[:self.k]

	def relative_pos_enc(self, xyz, p_idx, k_idx):
		# Assertion checks
		assert p_idx<len(xyz),\
		"point index = %d, out of range %d"%(p_idx,len(xyz_feat))
		assert k_idx<self.k,\
		"k_idx = %d is out of range, %d of K-NN."%(k_idx, self.k)

		center_point = xyz[p_idx]
		nn = xyz[k_idx]
		nn_diff = nn-center_point
		nn_diff_norm = linalg.norm(nn_diff).unsqueeze(-1)
		input = torch.cat((center_point, nn, nn_diff, nn_diff_norm))
		return self.MLP(input)

	def forward(self, xyz_feat, idx):
		nn_idx = self.gather_neighbor(xyz_feat, idx)

		r = torch.tensor([])
		for j in range(self.k):
			r = torch.cat((r, self.relative_pos_enc(
				xyz_feat[:,:3], idx, j)))
		r = r.reshape(-1, 3)
		f = torch.index_select(xyz_feat, 0, nn_idx)[:,:3]

		F = torch.cat((r,f), dim=1)
		return F

F = LocSE(k=8, d=3).forward(torch.randn(10, 6), 4)

class AttentionPool(nn.Module):
	def __init__(self, k, d_in, d_out):
		super().__init__()
		self.g = nn.Sequential(
			nn.Linear(
				in_features  = 2*d_in,
				out_features = 2*d_in,
				bias=False),
			nn.Softmax(dim=1))
		# self.SharedMLP = nn.Conv1d(
		# 	in_channels = 
		# )
		print(self.g)

	def forward(self, F):
		bs, n, k, d = F.size()
		f  = F.reshape((-1, k, d))
		print(f.size())

		exit()
		s = self.g(F)
		print(s, s.size())
		hadamard = F*s
		print(hadamard, hadamard.size())
		print(hadamard.T.sum(dim=1))

AttentionPool(k=8, d_in=3, d_out=10).forward(torch.randn(32, 200, 8, 3))

exit()


class DilatedResBlock(nn.Module):
	def __init__(self, d_in, d_out ):
		super().__init__()
		self.layers = nn.ModuleList([
			nn.Conv2d(
				in_channels  = d_in,
				out_channels = d_out//2,
				kernel_size  = [1,1],
				padding      = 'valid', ),
			nn.ReLU(),
			LocSE(k=3),
			AttentionPool(),
			LocSE(k=3),
			AttentionPool(),
			nn.Conv2d(
				in_channels  = d_in,
				out_channels = 2*d_out,
				kernel_size  = [1,1],
				padding      = 'valid', ),
		])
		self.shortcut = nn.ModuleList([
			nn.Conv2d(
				in_channels  = d_in,
				out_channels = 2*d_out,
				kernel_size  = [1,1],
				padding      = 'valid', ),
			nn.BatchNorm1d(2*d_out),
		])
		self.lrelu = nn.LeakyReLU()

	def forward(self, f):
		fpc = f
		for L in self.layers:
			fpc = L(fpc)

		sc = f
		for L in self.layers:
			sc = L(sc)

		return self.lrelu(sc+fpc)



N,d_in = 100, 3
DilatedResBlock(d_in, 6).forward(torch.randn(d_in, N, 3))
class RandLANet(nn.Module):
	def __init__(self,):
		super().__init__()

	def forward(self,):
		pass


