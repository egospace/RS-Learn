import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.modules.activation):
        super(Encoder, self).__init__()
        self.activation = activation
        self.miu = nn.Linear(in_features, out_features)
        self.sigma = nn.Linear(in_features, out_features)

    def forward(self, input_data: torch.Tensor, hidden_dimension: int) -> torch.Tensor:
        # 这里是生成对角的符合标准正态分布的矩阵还是生成形如h*h的标准正态分布矩阵？？？
        # return self.activation(self.miu(input_data) + self.sigma(input_data)
        #                        * torch.normal(mean=torch.zeros(hidden_dimension), std=torch.eye(hidden_dimension)))

        return self.activation(self.miu(input_data) + self.sigma(input_data)
                               * torch.randn_like(torch.eye(hidden_dimension)))


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
        self.dropout = nn.Dropout(drop_rate)
        self.encoder = Encoder(item_nums, hidden_dimension, nn.Identity())
        self.decoder = Decoder(hidden_dimension, item_nums, nn.Identity())

    def forward(self, ratting_vec):
        rwave_u = self.dropout(ratting_vec)
        z_u = self.encoder(rwave_u)
        x_u = self.decoder(z_u)
        return x_u
