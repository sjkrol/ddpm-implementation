
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 16, 8, 4]):
        super(UNet, self).__init__()

    def time_embedding(self, time_steps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        Returns the time embedding for a given timestep.
        @author: Stephen Krol

        :param time_steps: the timesteps to embed
        :type time_steps: torch.Tensor
        :param embedding_dim: the dimension of the time embedding
        :type embedding_dim: int

        :return: the time embedding
        :rtype: torch.Tensor
        """
