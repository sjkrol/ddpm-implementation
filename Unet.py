
import torch
import torch.nn as nn

import yaml

CONV_BLOCKS = 2
ATTEN_LAYER = 16
GROUP_NORM_GROUPS = 4
TIME_EMBEDDING_DIM = 16

class UNet(nn.Module):
    def __init__(self, original_channels: int=3, base_channels: int=128, channel_multipliers: list=[1, 2, 2, 2], num_res_blocks: int=2, in_resolution: int=32) -> None:
        """
        Initializes the UNet architecture for a diffusion model. The UNet consists of a series of down blocks, a middle block, and a series of up blocks.
        The down blocks consist of residual blocks and optional self-attention layers, with max pooling for downsampling. 
        The middle block consists of residual blocks and a self-attention layer. The up blocks consist of residual blocks and optional self-attention layers,
        with transposed convolutions for upsampling. Skip connections are used to connect the down blocks to the up blocks. This architecture is designed
        to follow the original UNet from the DDPM paper.
        @author: Stephen Krol

        :param original_channels: the number of channels in the input and output images (default: 3 for RGB)
        :type original_channels: int
        :param base_channels: the number of channels in the first layer of the UNet (default: 128)
        :type base_channels: int
        :param channel_multipliers: a list of multipliers for the number of channels in each subsequent block (default: [1, 2, 2, 2])
        :type channel_multipliers: list of int
        :param num_res_blocks: the number of residual blocks in each down and up block (default: 2)
        :type num_res_blocks: int
        :param in_resolution: the resolution of the input images (default: 32 for 32x32 images)
        :type in_resolution: int

        :return: None
        :rtype: None
        """
        super(UNet, self).__init__()

        # time embedding MLP
        self.time_MLP = TimeMLP(embedding_dim=TIME_EMBEDDING_DIM)

        # input convolution to get to the desired number of channels
        self.input_conv = nn.Conv2d(original_channels, base_channels, kernel_size=3, stride=1, padding=1)

        in_channels = base_channels # set initial in channels to base channels after the input convolution
        channels = [base_channels]  # to keep track of the number of channels at each block for skip connections

        # create down blocks with residual blocks and optional self-attention layers, using max pooling for downsampling.
        # each downblock consists of num_res_blocks residual blocks, followed by an optional self-attention layer if the 
        # resolution matches the attention layer, and then a max pooling layer for downsampling (except for the last block).
        self.down_blocks = nn.ModuleList()
        for i, channel_multiplier in enumerate(channel_multipliers):
            out_channels = base_channels * channel_multiplier

            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(in_channels, out_channels, time_embedding_dim=TIME_EMBEDDING_DIM))
                channels.append(out_channels)
                in_channels = out_channels

                if in_resolution == ATTEN_LAYER:
                    self.down_blocks.append(SelfAttention(in_channels))

            if i != len(channel_multipliers) - 1:
                self.down_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
                channels.append(out_channels)
                in_resolution = in_resolution // 2


        # create middle block
        # the middle block consists of a residual block, followed by a self-attention layer, and then another residual block. 
        # The number of channels in the middle block is the same as the output channels of the last down block.
        self.middle_blocks = nn.ModuleList()
        self.middle_blocks.append(ResBlock(in_channels, in_channels, time_embedding_dim=TIME_EMBEDDING_DIM))
        self.middle_blocks.append(SelfAttention(in_channels, is_middle_section=True))
        self.middle_blocks.append(ResBlock(in_channels, in_channels, time_embedding_dim=TIME_EMBEDDING_DIM))

        # create up blocks
        # each upblock consists of num_res_blocks residual blocks, followed by an optional self-attention layer if the 
        # resolution matches the attention layer, and then a conv transpose layer for upsampling (except for the last block).
        self.up_blocks = nn.ModuleList()
        for i, channel_multiplier in reversed(list(enumerate(channel_multipliers))):
            out_channels = base_channels * channel_multiplier

            # add residual blocks
            for _ in range(num_res_blocks + 1):  # +1 to account for the skip connection concatenation
                self.up_blocks.append(ResBlock(in_channels + channels.pop(), out_channels, time_embedding_dim=TIME_EMBEDDING_DIM  ))
                in_channels = out_channels

                # add self-attention layer if resolution matches
                if in_resolution == ATTEN_LAYER:
                    self.up_blocks.append(SelfAttention(in_channels))
            
            # upsample using conv transpose
            if i != 0:
                self.up_blocks.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2))
                in_resolution = in_resolution * 2

        # Final convolution to get the desired output channels
        self.final_conv = nn.Conv2d(in_channels, original_channels, kernel_size=1)
        self.output_norm = nn.GroupNorm(num_groups=original_channels, num_channels=original_channels)
        self.silu = nn.SiLU()
    
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
        hs = [x]

        # pass through down blocks
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
            
            if not isinstance(block, SelfAttention):
                hs.append(x)  # store the output of each block for skip connections
        
        for block in self.middle_blocks:
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
            else:
                x = block(x)

        # pass through up blocks
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                x = block(torch.concat([x, hs.pop()], dim=1), t_emb)
            else:
                x = block(x)

        # pass through final convolution to get the desired output channels
        return self.silu(self.output_norm(self.final_conv(x)))


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

    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int) -> None:
        """
        Initializes a residual block with time embedding.
        @author: Stephen Krol

        :param in_channels: the number of input channels
        :type in_channels: int
        :param out_channels: the number of output channels
        :type out_channels: int
        :param time_embedding_dim: the dimension of the time embedding
        :type time_embedding_dim: int

        :return: None
        :rtype: None
        """
        super(ResBlock, self).__init__()

        self.norm1 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=in_channels)
        self.norm2 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=out_channels)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

        # first convolutional block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # second convolutional block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # time projection layer
        self.time_proj = nn.Linear(time_embedding_dim, out_channels)

        # for skip connection, if the number of input channels is different from the number of output channels, we need to project the input to the correct number of channels
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
        h = self.silu(self.norm2(h))

        # add dropout
        h = self.dropout(h)

        # ensure x has the same number of channels as h for the skip connection
        if x.shape[1] != h.shape[1]:
            x = self.x_conv(x)

        return self.conv2(h) + x

    
class TimeMLP(nn.Module):

    def __init__(self, embedding_dim: int) -> None:
        """
        Initializes the time MLP.
        @author: Stephen Krol

        :param embedding_dim: the dimension of the time embedding
        :type embedding_dim: int

        :return: None
        :rtype: None
        """
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

    def __init__(self, in_channels: int, is_middle_section: bool = False) -> None:
        """
        Initializes the self-attention layer.
        @author: Stephen Krol

        :param in_channels: the number of input channels
        :type in_channels: int
        :param is_middle_section: whether this layer is in the middle section of the UNet
        :type is_middle_section: bool

        :return: None
        :rtype: None
        """
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(in_channels, in_channels)
        self.W_k = nn.Linear(in_channels, in_channels)
        self.W_v = nn.Linear(in_channels, in_channels)

        self.norm = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=in_channels)

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

        h = self.norm(x)
        # assert C == 16 or self.is_middle_section, "Self-attention is only implemented for C=16 or within middle section in this UNet architecture."
        h = h.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # compute query, key, and value matrices
        q, k, v = self.W_q(h), self.W_k(h), self.W_v(h)  # (B, H*W, C)

        # compute attention weights
        atten_weights = torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5)  # (B, H*W, H*W)
        atten_scores = torch.softmax(atten_weights, dim=-1)  # (B, H*W, H*W)

        # compute output
        out = torch.bmm(atten_scores, v)  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        return out + x

        
if __name__ == "__main__":
    
    yaml_path = "/Users/sjkro1/Documents/Personal/coding/DiffusionImplementation/config.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    model = UNet()  

    x = torch.randn(4, 3, 32, 32)  # Example input image
    time_steps = torch.tensor([10])  # Example time step

    output = model(x, time_steps)
    print("Output shape:", output.shape)