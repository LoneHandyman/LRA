import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.complex_scan import complex_scan

from typing import Type

from einops import rearrange, repeat

class GlobalConv(nn.Module):
    def __init__(self, d_model: int, d_conv: int) -> None:
        super().__init__()

        self.scale = 1 / (d_model ** 0.5)

        self.from_w = nn.Linear(d_model, d_model, bias=False)
        self.conv = nn.Conv1d(d_model, d_model, d_conv, 
                              padding=d_conv // 2)
        
    def forward(self, x_to: torch.Tensor, x_from: torch.Tensor) -> torch.Tensor:
        XF = rearrange(x_from, "b n d -> b d n")
        cxf = F.silu(self.conv(XF))
        cxf = rearrange(cxf, "b d n -> b n d")
        
        s = x_to * F.softmax(self.from_w(cxf) * self.scale, dim=-2)
        glob_s = torch.sum(s, dim=-2)

        glob_s = repeat(glob_s, "b d -> b n d", n=x_to.size(-2))

        return glob_s

class Summer(nn.Module):

    def __init__(self, d_model: int, d_conv: int) -> None:
        super().__init__()

        self.d_model = d_model

        self.summ = GlobalConv(d_model, d_conv)

        self.in_proj = nn.Linear(self.d_model, self.d_model*4)
        self.mid_proj = nn.Linear(self.d_model, self.d_model*2)
        self.out_proj = nn.Linear(2*self.d_model, self.d_model)

        nu_log, theta_log, gamma_log = self.initializer()
        self.nu_log = nn.Parameter(nu_log, requires_grad=True)
        self.theta_log = nn.Parameter(theta_log, requires_grad=True)
        self.gamma_log = nn.Parameter(gamma_log, requires_grad=True)

        self.dropout = nn.Dropout(p=0.2)
        self.l_norm = nn.LayerNorm(self.d_model*2, elementwise_affine=False)

    def initializer(self):
        r_min, r_max = 0.9, 0.999
        u1 = np.random.random(self.d_model)
        u2 = np.random.random(self.d_model)
        nu_log = np.log(
            -0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )
        theta_log = np.log(u2 * np.pi * 2)
        gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
        
        return torch.Tensor(nu_log), torch.Tensor(theta_log), torch.Tensor(gamma_log)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.in_proj(x)
        qo, g  = u.chunk(2,dim=-1)
        q, o = qo.chunk(2,dim=-1)

        nu = torch.exp(-torch.exp(self.nu_log))
        theta = torch.exp(self.theta_log) 
        gamma = torch.exp(self.gamma_log)

        f_real = nu * torch.cos(theta)
        f_imag = nu * torch.sin(theta)

        v = self.mid_proj(o * self.summ(q, q))
        
        input_real, input_imag = v.chunk(2, dim=-1)
        input_real = gamma[None, None, :] * input_real
        input_imag = gamma[None, None, :] * input_imag        
        
        f_real = f_real[None, None, :].expand_as(input_real)
        f_imag = f_imag[None, None, :].expand_as(input_real)
    
        output_real, output_imag = complex_scan(
            input_real.contiguous(), input_imag.contiguous(),
            f_real.contiguous(), f_imag.contiguous()
        )

        return self.out_proj(
            self.l_norm(
                self.dropout(
                    torch.cat([output_real, output_imag], dim=-1) * F.silu(g)
                )
            )
        )
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: int=0.1) -> None:
        super(FeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, hidden)
        self.actv = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(hidden, d_model)
        
    def forward(self, x: torch.Tensor):
        x = self.w1(x)
        x = self.actv(x)
        x = self.dropout(x)
        x = self.w2(x)

        return x
    
class SummeRBlock(nn.Module):
    def __init__(self, d_model: int, d_conv: int, hidden: int, dropout: int) -> None:
        super().__init__()
        self.token_mixer = Summer(d_model, d_conv)
        self.channel_mixer = FeedForward(d_model, hidden, dropout)

        self.l_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x: torch.Tensor):
        x = x + self.token_mixer(x)
        x = self.l_norm(x + self.channel_mixer(x))

        return x

class SummeRNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        assert config["embedding_dim"] == config["transformer_dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)
        self.dropout = nn.Dropout(p = config["dropout_prob"])

        self.blocks = nn.ModuleList([SummeRBlock(config["transformer_dim"],
                                                 config["d_conv"], 
                                                 config["transformer_hidden_dim"],
                                                 config["dropout_prob"]) 
                                                 for _ in config["num_layers"]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.word_embeddings(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        return x