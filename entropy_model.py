import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

import torchac


class RoundNoGradient(torch.autograd.Function):
    """ TODO: check. """
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-9)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        try:
            grad1[x<1e-9] = 0
        except RuntimeError:
            print("ERROR! grad1[x<1e-9] = 0")
            grad1 = g.clone()
        pass_through_if = np.logical_or(x.cpu().detach().numpy() >= 1e-9, g.cpu().detach().numpy()<0.0)
        t = torch.Tensor(pass_through_if+0.0).to(grad1.device)

        return grad1*t


class EntropyBottleneck(nn.Module):
    """The layer implements a flexible probability density model to estimate
    entropy of its input tensor, which is described in this paper:
    >"Variational image compression with a scale hyperprior"
    > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
    > https://arxiv.org/abs/1802.01436"""
    
    def __init__(self, channels, init_scale=8, filters=(3,3,3)):
        """create parameters.
        """
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = 1e-9
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        self._channels = channels
        self.ASSERT = False
        # build.
        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))
        # Create variables.
        self._matrices = nn.ParameterList([])
        self._biases = nn.ParameterList([])
        self._factors = nn.ParameterList([])

        for i in range(len(self._filters) + 1):
            #
            self.matrix = Parameter(torch.FloatTensor(channels, filters[i + 1], filters[i]))
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix.data.fill_(init_matrix)
            self._matrices.append(self.matrix)
            #
            self.bias = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            init_bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, self.bias.size()))
            self.bias.data.copy_(init_bias)# copy or fill?
            self._biases.append(self.bias)
            #       
            self.factor = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            self.factor.data.fill_(0.0)
            self._factors.append(self.factor)

    def _logits_cumulative(self, inputs):
        """Evaluate logits of the cumulative densities.
        
        Arguments:
        inputs: The values at which to evaluate the cumulative densities,
            expected to have shape `(channels, 1, batch)`.

        Returns:
        A tensor of the same shape as inputs, containing the logits of the
        cumulatice densities evaluated at the the given inputs.
        """
        logits = inputs
        for i in range(len(self._filters) + 1):
            matrix = torch.nn.functional.softplus(self._matrices[i])
            logits = torch.matmul(matrix, logits)
            logits += self._biases[i]
            factor = torch.tanh(self._factors[i])
            logits += factor * torch.tanh(logits)
        
        return logits

    def _quantize(self, inputs, mode):
        """Add noise or quantize."""
        if mode == "noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            return inputs + noise
        if mode == "symbols":
            return RoundNoGradient.apply(inputs)

    def _likelihood(self, inputs):
        """Estimate the likelihood.
        inputs shape: [points, channels]
        """
        # reshape to (channels, 1, points)
        inputs = inputs.permute(1, 0).contiguous()# [channels, points]
        shape = inputs.size()# [channels, points]
        inputs = inputs.view(shape[0], 1, -1)# [channels, 1, points]
        inputs = inputs.to(self.matrix.device)
        # Evaluate densities.
        lower = self._logits_cumulative(inputs - 0.5)
        upper = self._logits_cumulative(inputs + 0.5)
        sign = -torch.sign(torch.add(lower, upper)).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        # reshape to (points, channels)
        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0)

        return likelihood

    def forward(self, inputs, quantize_mode="noise"):
        """Pass a tensor through the bottleneck.
        """
        if quantize_mode is None: outputs = inputs
        else: outputs = self._quantize(inputs, mode=quantize_mode)
        likelihood = self._likelihood(outputs)
        likelihood = Low_bound.apply(likelihood)

        return outputs, likelihood

    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    @torch.no_grad()
    def compress(self, inputs):
        # quantize
        values = self._quantize(inputs, mode="symbols")
        # get symbols
        min_v = values.min().detach().float()
        max_v = values.max().detach().float()
        symbols = torch.arange(min_v, max_v+1)
        symbols = symbols.reshape(-1,1).repeat(1, values.shape[-1])# (num_symbols, channels)
        # get normalized values
        values_norm = values - min_v
        min_v, max_v = torch.tensor([min_v]), torch.tensor([max_v])
        values_norm = values_norm.to(torch.int16)

        # get pmf
        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        pmf = pmf.permute(1,0)# (channels, num_symbols)

        # get cdf
        cdf = self._pmf_to_cdf(pmf)
        # arithmetic encoding
        out_cdf = cdf.unsqueeze(0).repeat(values_norm.shape[0], 1, 1).detach().cpu()
        strings = torchac.encode_float_cdf(out_cdf, values_norm.cpu(), check_input_bounds=True)

        return strings, min_v.cpu().numpy(), max_v.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, min_v, max_v, shape, channels):
        # get symbols
        symbols = torch.arange(min_v, max_v+1)
        symbols = symbols.reshape(-1,1).repeat(1, channels)

        # get pmf
        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        pmf = pmf.permute(1,0)
        # get cdf
        cdf = self._pmf_to_cdf(pmf)
        # arithmetic decoding
        out_cdf = cdf.unsqueeze(0).repeat(shape[0], 1, 1).detach().cpu()
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        values += min_v

        return values