# 2020.02.13 compression network
import sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

from models.AutoEncoder import Encoder, Decoder
from models.EntropyModel import EntropyBottleneck


class PCC(nn.Module):
	"""
	Encoder
	"""

	def __init__(self, channels=8):
		nn.Module.__init__(self)
		# self.nchannels=channels
		self.encoder = Encoder(channels=channels, block_layers=3, block='InceptionResNet')
		self.decoder = Decoder(channels=channels, block_layers=3, block='InceptionResNet')
		self.entropy_bottleneck = EntropyBottleneck(channels)

	def forward(self, x, target_format, adaptive, training, device):
		ys = self.encoder(x)

		# new sparse tensor.
		y = ME.SparseTensor(ys[0].F, coordinates=ys[0].C, tensor_stride=8, device=device)
		# add noise to feature
		feats_tilde, likelihood = self.entropy_bottleneck(y.F, training, device)
		y_tilde = ME.SparseTensor(feats_tilde, coordinate_map_key=y.coordinate_map_key, coordinate_manager=y.coordinate_manager, device=device)

		cm = y_tilde.coordinate_manager
		# target_key = cm.create_coords_key(
		# 	x.C,
		# 	force_creation=True,
		# 	allow_duplicate_coords=True)
		# TODO from v0.4 to v0.5
		target_map_key = cm.insert_and_map(
			x.C,
			tensor_stride=1)

		if target_format == 'key':
			out, out_cls, targets, keeps = self.decoder(y_tilde, target_label=target_map_key, adaptive=adaptive, training=training)
		elif target_format == 'sp_tensor':
			out, out_cls, targets, keeps = self.decoder(y_tilde, target_label=ys[1:]+[x], adaptive=adaptive, training=training)
		else:
			print('Target Label Format Error!')
			sys.exit(0)

		return ys, likelihood, out, out_cls, targets, keeps

if __name__ == '__main__':
	pcc = PCC(channels=8)
	print(pcc)

