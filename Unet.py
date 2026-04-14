
import torch
import torch.nn as nn

CONV_BLOCKS = 2
ATTEN_LAYER = 16
GROUP_NORM_GROUPS = 4
TIME_EMBEDDING_DIM = 16

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 16, 8, 4]):
        super(UNet, self).__init__()

        self.time_MLP = TimeMLP(embedding_dim=TIME_EMBEDDING_DIM)

        in_feature = features[0]

        # input convolution to get to the desired number of channels
        self.input_conv = nn.Conv2d(in_channels, in_feature, kernel_size=3, stride=1, padding=1)

        # create down blocks
        self.down_blocks = nn.ModuleList()
        for feature in features:
            self.down_blocks.append(ResBlock(in_feature, feature, time_embedding_dim=TIME_EMBEDDING_DIM))
            self.down_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_feature = feature


        # create middle block


        # Final convolution to get the desired output channels
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.
        @author: Stephen Krol

        :param x: the input tensor
        :type x: torch.Tensor
        :param time_steps: the timesteps for time embedding
        :type time_steps: torch.Tensor

        :return: the output tensor
        :rtype: torch.Tensor
        """

        # compute time embedding and pass through MLP
        t_emb = self.time_embedding(time_steps, TIME_EMBEDDING_DIM)
        t_emb = self.time_MLP(t_emb)

        # pass through input convolution to get to the desired number of channels
        x = self.input_conv(x)

        # pass through down blocks
        for block in self.down_blocks:
            print(f"Block: {block.__class__.__name__}")
            print(x.shape)
            print()
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
            else:
                x = block(x)


        return x


    def time_embedding(self, 
                       time_steps: torch.Tensor, 
                       embedding_dim: int) -> torch.Tensor:
        """
        Returns the time embedding for a given timestep.
        The sine and cosine terms are concatenated rather than interleaved because this preserves the same information in a simpler layout.
        This embedding is typically passed through a shared MLP and then linearly projected to the channel size needed by each residual block.
        @author: Stephen Krol

        :param time_steps: the timesteps to embed
        :type time_steps: torch.Tensor
        :param embedding_dim: the dimension of the time embedding
        :type embedding_dim: int

        :return: the time embedding
        :rtype: torch.Tensor
        """

        half_dim = embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = time_steps[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(ResBlock, self).__init__()

        print(f"ResBlock: in_channels={in_channels}, group norm groups={GROUP_NORM_GROUPS}")

        self.norm1 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=in_channels)
        self.norm2 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=out_channels)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

        # first convolutional block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # if the number of channels matches the attention layer, add a self-attention layer
        if out_channels == ATTEN_LAYER:
            self.attention = SelfAttention(out_channels)

        # second convolutional block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # time projection layer
        self.time_proj = nn.Linear(time_embedding_dim, out_channels)

        # self.x_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.x_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()



    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the down block. Note that this expects the input tensor x to be unnormalised and not activated, 
        as the first step in the block is to normalise and activate it. The time embedding is projected and added to the output 
        of the first convolutional block, and then passed through a self-attention layer if the number of channels matches the attention layer. 
        Finally, a skip connection adds the input tensor to the output of the second convolutional block.
        @author: Stephen Krol

        :param x: the input tensor
        :type x: torch.Tensor
        :param t_emb: the time embedding tensor
        :type t_emb: torch.Tensor

        :return: the output tensor
        :rtype: torch.Tensor
        """

        # project time embedding and reshape for broadcasting
        t_emb = self.time_proj(t_emb)[:, :, None, None]

        # assumes input is not normalised or activated
        h = self.silu(self.norm1(x))
        
        # pass through conv net and add time embedding
        h = self.conv1(h)
        h = h + t_emb

        # add self-attention if the number of channels matches the attention layer
        if hasattr(self, 'attention'):
            h = self.attention(h)

        h = self.silu(self.norm2(h))

        # add dropout
        h = self.dropout(h)

        # ensure x has the same number of channels as h for the skip connection
        if x.shape[1] != h.shape[1]:
            x = self.x_conv(x)

        return self.conv2(h) + x


    
class TimeMLP(nn.Module):

    def __init__(self, embedding_dim):
        super(TimeMLP, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(embedding_dim * 4, embedding_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the time MLP.
        @author: Stephen Krol

        :param t: the time embedding tensor
        :type t: torch.Tensor

        :return: the output tensor
        :rtype: torch.Tensor
        """

        return self.silu(self.linear2(self.silu(self.linear1(t))))

class SelfAttention(nn.Module):

    def __init__(self, in_channels, is_middle_section=False):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(in_channels, in_channels)
        self.W_k = nn.Linear(in_channels, in_channels)
        self.W_v = nn.Linear(in_channels, in_channels)

        self.is_middle_section = is_middle_section

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs self-attention on the input tensor.
        @author: Stephen Krol

        :param x: the input tensor
        :type x: torch.Tensor

        :return: the output tensor after self-attention
        :rtype: torch.Tensor
        """

        # reshape x to (B, H*W, C) for self-attention (C = d)
        B, C, H, W = x.shape
        assert C == 16 or self.is_middle_section, "Self-attention is only implemented for C=16 or within middle section in this UNet architecture."
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # compute query, key, and value matrices
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)  # (B, H*W, C)

        # compute attention weights
        atten_weights = torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5)  # (B, H*W, H*W)
        atten_scores = torch.softmax(atten_weights, dim=-1)  # (B, H*W, H*W)

        # compute output
        out = torch.bmm(atten_scores, v)  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        return out

        


if __name__ == "__main__":
    model = UNet()  

    x = torch.randn(1, 3, 64, 64)  # Example input image
    time_steps = torch.tensor([10])  # Example time step

    output = model(x, time_steps)
    print("Output shape:", output.shape)