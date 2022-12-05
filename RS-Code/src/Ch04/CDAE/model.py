import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, user_num: int, activation: nn.modules.activation):
        super(Encoder, self).__init__()
        self.activation = activation
        self.W = nn.Parameter(
            torch.rand(out_features, in_features) * 4 * math.sqrt(6 / (in_features + out_features)))
        self.b_h = nn.Parameter(torch.zeros(out_features))
        self.U = nn.Embedding(user_num, out_features).from_pretrained(
            torch.rand(user_num, out_features) * 4 * math.sqrt(6 / (in_features + out_features)))

    def forward(self, input_data: torch.Tensor, user_id: torch.Tensor) -> torch.Tensor:
        # return self.activation(F.linear(input_data, self.W, self.b_h) + self.U(user_id).squeeze(dim=1))
        return self.activation(F.linear(input_data, self.W))


class Decoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.modules.activation):
        super(Decoder, self).__init__()
        self.activation = activation
        self.V = nn.Parameter(torch.rand(out_features, in_features) * 4 * math.sqrt(6 / (in_features + out_features)))
        self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.activation(F.linear(input_data, self.V, self.b))


class CDAE(nn.Module):
    def __init__(self, user_nums: int, item_nums: int, hidden_dimension: int, drop_rate: float):
        super(CDAE, self).__init__()
        self.rwave_u = None
        self.denoiser = nn.Dropout(drop_rate)
        self.encoder = Encoder(item_nums, hidden_dimension, user_nums, nn.Identity())
        self.decoder = Decoder(hidden_dimension, item_nums, nn.Identity())

    def forward(self, uid, ratting_vec):
        rwave_u = self.denoiser(ratting_vec)
        self.rwave_u = rwave_u
        z_u = self.encoder(rwave_u, uid)
        r_hat_u = self.decoder(z_u)
        return r_hat_u
