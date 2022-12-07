import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.modules.activation):
        super(Encoder, self).__init__()
        self.activation = activation
        self.miu = nn.Linear(in_features, out_features)
        self.var = nn.Linear(in_features, out_features)

    def forward(self, input_data: torch.Tensor, hidden_dimension: int):
        # 这里是生成对角的符合标准正态分布的矩阵还是生成形如h*h的标准正态分布矩阵？？？
        miu = self.miu(input_data)
        log_var = self.var(input_data)
        if self.training:
            std = torch.exp(0.5 * log_var)
            # z_u = self.activation(miu + torch.matmul(std, torch.normal(mean=torch.zeros(hidden_dimension), std=torch.eye(hidden_dimension))))
            z_u = self.activation(miu + torch.mul(std, torch.randn_like(std)))
        else:
            z_u = miu
        return z_u, miu, log_var


class Decoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.modules.activation):
        super(Decoder, self).__init__()
        self.activation = activation
        self.Xu = nn.Linear(in_features, out_features)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.activation(self.Xu(input_data))


class VAE(nn.Module):
    def __init__(self, item_nums: int, hidden_dimension: int, drop_rate: float):
        super(VAE, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.dropout = nn.Dropout(drop_rate)
        self.encoder = Encoder(item_nums, hidden_dimension, nn.Identity())
        self.decoder = Decoder(hidden_dimension, item_nums, nn.Identity())

    def forward(self, ratting_vec):
        rwave_u = self.dropout(ratting_vec)
        z_u, miu, log_var = self.encoder(rwave_u, self.hidden_dimension)
        x_u = self.decoder(z_u)
        return x_u, miu, log_var
