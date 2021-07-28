import torch
import MinkowskiEngine as ME

from autoencoder import Encoder, Decoder
from entropy_model import EntropyBottleneck


class PCCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, training=True):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x] 
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model
        y_q, likelihood = self.get_likelihood(y, 
            quantize_mode="noise" if training else "symbols")

        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        return {'out':out,
                'out_cls_list':out_cls_list,
                'prior':y_q, 
                'likelihood':likelihood, 
                'ground_truth_list':ground_truth_list}

if __name__ == '__main__':
    model = PCCModel()
    print(model)

