import torch
from torch import nn
from torch import linalg

print("CUDA Available: ", torch.cuda.is_available())
class SharedMLP(nn.Module):
	def __init__(self, in_chan, out_chan, transpose = False):
		super().__init__()
		if not transpose:
			self.conv = nn.Conv2d(in_chan, out_chan,
				kernel_size=[1,1], padding='valid')
		else:
			self.conv = nn.ConvTranspose2d(in_chan, out_chan,
				kernel_size=[1, 1])
	def forward(self, x):
		# Input : [0:B, 1:N, 2:in_chan ]
		# Output: [0:B, 1:N, 2:out_chan]
		x = x.permute(0,2,1).unsqueeze(-1)
		return self.conv(x).squeeze().permute(0,2,1)

# print("From SharedMLP: ",
# 	SharedMLP(6, 8)(
# 		torch.randn(32, 100, 6)
# 	).size())
# exit()

class LocSE(nn.Module):
	def __init__(self, d_in, k):
		super().__init__()
		self.k = k
		self.d_in = d_in
		self.mlp = nn.Sequential(
			# concatenation of [3,3,3,1]=10 as input
			SharedMLP(10, d_in),
			nn.ReLU(), )

	def gather_neighbor(self, xyz, feat, idx):
		# Input
		# xyz : [B, N, 3]	x,y,z co-ordinates for N points
		# feat: [B, N, d_in] 	d_in features of N points
		# idx : [B, N, K]	indices of K near N points

		# Do something here
		B,N,d_in=feat.size()

		# Output
		# p: [B, N, K, 3]	x,y,x of K-neighbors for each N points
		# f: [B, N, K, d_in]	d_in features of above points
		return (
			torch.randn(B, N, self.k, 3),
			torch.randn(B, N, self.k, self.d_in)
		)

		assert self.k<=len(xyz_feat),\
		"k = %d should be less than n = %d"%(self.k, len(xyz_feat))

		xyz = xyz_feat[:,:3]
		p = xyz_feat[p_idx,:3]
		diff = xyz-p
		norm = linalg.norm(xyz-p, dim=1)
		_, neighbor = norm.sort()
		return neighbor[:self.k]


	def relative_pos_enc(self, xyz, idx):
		# Input
		# xyz: [B, N, K, 3]	x,y,z co-ordinates for N points
		# idx: [B, N, K]	indices of k nearest neighbor of N points

		# Do something
		B,N,_,d_in=xyz.size()

		# Output
		# r: [B, N, K, d_in]
		return torch.randn(B, N, self.k, self.d_in)

		for p_idx in idx:
			center_point = xyz[:, :, p_idx]
			print(center_point.size())
			exit()
			nn = xyz[k_idx]
			nn_diff = nn-center_point
			nn_diff_norm = linalg.norm(nn_diff).unsqueeze(-1)
			input = torch.cat((center_point, nn, nn_diff, nn_diff_norm))
		return self.MLP(input)
		# Returns a batch B of N points of size 10, r: [B, N, K, d_out]

	def forward(self, xyz, feat, idx):
		# Input
		# xyz:  [B, N, 3]	x,y,z co-ordinates for N points
		# feat: [B, N, d_in]	d features of N points
		# idx:  [B, N, k]	k indices for N neighbor points

		# Gather neighbor
		p,f = self.gather_neighbor(xyz, feat, idx)
		print("After gathering neighbors: ", p.size(),f.size())

		# Relative position encoding
		# Input : xyz=[B, N, K, 3], idx=[B, N, K]
		# Output: [B, N, K, d_in]
		r = self.relative_pos_enc(p, idx)
		print("Relative pos enc: ", r.size())
		print("Features        : ", f.size())

		# Concatenation
		# Input : r=[B, N, K, d], f=[B, N, K, d]
		# Output: rf=[B, N, K, 2d]
		F = torch.cat((r,f), dim=3)
		print("After concatenation: ", F.size());

		# Returns a batch of B points with
		# feat: [B, N, K, d_in]
		return F

B,N,k = 32, 100, 8
LocSE(d_in=3, k=k)(
	torch.randn(B, N, 3),
	torch.randn(B, N, 3),
	torch.tensor(torch.rand(B, N, k)*k, dtype=torch.int8)
)

class AttentionPool(nn.Module):
	def __init__(self, d_in, d_out):
		super().__init__()
		self.g = nn.Sequential(
			nn.Linear(d_in, d_in, bias=False),
			nn.Softmax(dim=1),)
		self.mlp= SharedMLP(d_in, d_out)

	def forward(self, F):
		# Input
		# F: [B, N, K, d_in]

		print("Input to AttentionPool %s: %s" %(self.g, F.size()))
		scores = self.g(F)
		# Output
		# F: [B, N, d_out]
		return self.mlp((scores*F).sum(dim=2))

# print("From AttentionPool: ",
# 	AttentionPool(6, 10)(
# 		torch.randn(B, N, k, 6)
# 	).size())
# exit()

class DilatedResBlock(nn.Module):
	def __init__(self, d_in, d_out, k):
		super().__init__()
		self.k = k
		self.layers = nn.ModuleList([
			SharedMLP(d_in, d_out//2),
			nn.LeakyReLU(.2),
			LocSE(d_out//2, k),
			AttentionPool(d_out, d_out//2),
			LocSE(d_out, k),
			AttentionPool(d_out, d_out),
			SharedMLP(d_out, 2*d_out),
		])
		self.shortcut = SharedMLP(d_in, 2*d_out)
		self.lrelu = nn.LeakyReLU()

	def forward(self, xyz, feat):
		# Input
		# xyz : [B, N, 3]
		# feat: [B, N, d_in]

		print(xyz.size(), feat.size())
		idx = [i for i in range(10)]
		sc = self.shortcut(feat)
		for L in self.layers:
			if L.__class__.__name__ == 'LocSE':
				feat = L(xyz, feat, idx)
			else:
				feat = L(feat)

			print("%-20s%s"%(L.__class__.__name__, feat.size()),
				flush=True)

		# return self.lrelu(sc+feat)

DilatedResBlock(d_in=3, d_out=10, k=10)(
	torch.rand(32,100,3),
	torch.randn(32,100,3))
exit()
N,d_in = 100, 3
DilatedResBlock(d_in, 6).forward(torch.randn(d_in, N, 3))
class RandLANet(nn.Module):
	def __init__(self,):
		super().__init__()

	def forward(self,):
		pass


