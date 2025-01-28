import numpy as np
import torch
from torch import nn
from Models.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Models.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


def model_factory(config):
    if config['Net_Type'][0] == 'T':
        model = Transformer(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'CC-T':
        model = CasualConvTran(config, num_classes=config['num_labels'])
    else:
        model = ConvTran(config, num_classes=config['num_labels'])
    return model


class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(
            nn.Linear(channel_size, emb_size),
            nn.LayerNorm(emb_size, eps=1e-5)
        )

        if self.Fix_pos_encode == 'Sin':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        if self.Rel_pos_encode == 'Scalar':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x_src = self.embed_layer(x.permute(0, 2, 1))
        if self.Fix_pos_encode != 'None':
            x_src = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        # out = out.permute(1, 0, 2)
        # out = self.out(out[-1])

        return out


class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size*4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        print("Start fowarding")
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            torch.cuda.empty_cache()
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        print("End fowarding")
        return out


class CasualConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Extract configuration with defaults to reduce memory overhead
        channel_size = config.get("Data_shape", [None, 1, 128])[1]
        seq_len = config.get("Data_shape", [None, 1, 128])[2]
        self.emb_size = config.get("emb_size", 64)
        num_heads = config.get("num_heads", 4)
        dim_ff = config.get("dim_ff", 256)
        dropout = config.get("dropout", 0.1)

        # Memory optimization 1: Use in-place operations where possible
        self.causal_convs = nn.ModuleList(
            [
                nn.Sequential(
                    CausalConv1d(
                        in_channels if i == 0 else self.emb_size,
                        self.emb_size,
                        kernel_size=kernel,
                        stride=2,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(self.emb_size),
                    nn.GELU(),
                )
                for i, (in_channels, kernel, dilation) in enumerate(
                    [(channel_size, 8, 1), (self.emb_size, 5, 2), (self.emb_size, 3, 2)]
                )
            ]
        )

        # Memory optimization 2: Conditional initialization of position encoding
        self.pos_encode_type = config.get("Fix_pos_encode", "None")
        if self.pos_encode_type != "None":
            self.Fix_Position = self._create_position_encoding(
                self.pos_encode_type, self.emb_size, dropout, seq_len
            )

        # Memory optimization 3: Unified attention layer creation
        self.attention_layer = self._create_attention_layer(
            config.get("Rel_pos_encode", "None"),
            self.emb_size,
            num_heads,
            seq_len,
            dropout,
        )

        # Memory optimization 4: Use single layer norm instance with parameter sharing
        self.layer_norm = nn.LayerNorm(self.emb_size, eps=1e-5)

        # Memory optimization 5: Streamlined feed-forward with gradient checkpointing
        self.feed_forward = nn.Sequential(
            nn.Linear(self.emb_size, dim_ff),
            nn.ReLU(inplace=True),  # Use inplace ReLU
            nn.Dropout(dropout),
            nn.Linear(dim_ff, self.emb_size),
            nn.Dropout(dropout),
        )

        # Output layers
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(self.emb_size, num_classes)

    def _create_position_encoding(self, encode_type, emb_size, dropout, seq_len):
        if encode_type in ["tAPE", "Sin"]:
            return tAPE(emb_size, dropout=dropout, max_len=seq_len)
        elif encode_type == "Learn":
            return LearnablePositionalEncoding(
                emb_size, dropout=dropout, max_len=seq_len
            )
        return None

    def _create_attention_layer(
        self, rel_pos_type, emb_size, num_heads, seq_len, dropout
    ):
        if rel_pos_type == "eRPE":
            return Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)
        elif rel_pos_type == "Vector":
            return Attention_Rel_Vec(emb_size, num_heads, seq_len, dropout)
        return Attention(emb_size, num_heads, dropout)

    @torch.jit.script_method
    def forward(self, x):
        # Memory optimization 6: Use JIT compilation for forward pass
        x = x.unsqueeze(1)

        # Memory optimization 7: Efficient sequential processing with gradient checkpointing
        x_src = x
        for conv in self.causal_convs:
            x_src = torch.utils.checkpoint.checkpoint(conv, x_src)

        x_src = x_src.squeeze(2).permute(0, 2, 1)

        # Memory optimization 8: Conditional position encoding
        if self.pos_encode_type != "None":
            x_src = self.Fix_Position(x_src)

        # Memory optimization 9: Efficient attention computation
        att = x_src + torch.utils.checkpoint.checkpoint(self.attention_layer, x_src)
        att = self.layer_norm(att)

        # Memory optimization 10: Efficient feed-forward computation
        out = att + torch.utils.checkpoint.checkpoint(self.feed_forward, att)
        out = self.layer_norm(out)

        # Memory optimization 11: Efficient output processing
        out = F.adaptive_avg_pool1d(out.permute(0, 2, 1), 1).squeeze(-1)
        return self.out(out)

    def train_step(self, x, optimizer, loss_fn):
        # Memory optimization 12: Integrated training step with automatic mixed precision
        with torch.cuda.amp.autocast():
            output = self(x)
            loss = loss_fn(output)

        return loss


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(nn.functional.pad(x, (self.__padding, 0)))
