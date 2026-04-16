import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# ------------------------------------------------------------
# Time embedding with FiLM (scale + shift + optional gate)
# ------------------------------------------------------------
# Transformer positional encoding and MLP embedding for timestep
class TimestepEmbedder(nn.Module):
    def __init__(self, max_seq_len:int, temb_dim:int=128, max_period:float=10000.0, mlp_ratio:int=2, act_layer=nn.SiLU):
        # max_seq_len: equals to time steps in diffusion process (used for pre-calculation required by nn.Embedding)
        super().__init__()
        assert temb_dim % 2 == 0, 'embedding dim must be even'

        pe = torch.zeros(max_seq_len, temb_dim)
        # position (time) [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        
        # calculate w = 1 / [10000^(2i/temb_dim)]
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, temb_dim, 2, dtype=torch.float32) / temb_dim)
        
        # position encoding: [sin(wt), cos(wt), sin(wt+t), cos(wt+t), ...]
        pe[:, 0::2] = torch.sin(freqs * position)
        pe[:, 1::2] = torch.cos(freqs * position)
        
        self.embedding = nn.Embedding(max_seq_len, temb_dim)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

        self.time_mlp = nn.Sequential(
                nn.Linear(temb_dim, temb_dim*mlp_ratio),
                act_layer(),
                nn.Linear(temb_dim*mlp_ratio, temb_dim))

    def forward(self, t:torch.Tensor):
        # embed t to a temb_dim-dimensional vector: (batch_size,) -> (batch_size, temb_dim)
        # t must be torch.long or int required by nn.Embedding
        t = t.long() if t.dtype != torch.long else t
        temb = self.time_mlp(self.embedding(t))
        return temb


# FiLM layer to generate scale, shift, and gate from time embedding (usually after the norm layer)
class FiLM_Layer_ada(nn.Module):
    def __init__(self, ch_conv:int, temb_dim:int=128, num_affine:int=2, act_layer=nn.SiLU):
        super().__init__()
        self.num_affine = num_affine
        self.mlp = nn.Sequential(
            nn.Linear(temb_dim, ch_conv*num_affine),
            act_layer(),
            nn.Linear(ch_conv*num_affine, ch_conv*num_affine)
        )

    def forward(self, temb:torch.Tensor) -> tuple:
        """
        Input: t (batch_size, temb_dim)
        Output: gate (optional), scale, shift (batch_size, ch_conv)
        """
        x = self.mlp(temb)  # (B, NC)
        if self.num_affine == 2:
            scale, shift = x.chunk(2, dim=-1)
            gate = torch.ones_like(scale)
        elif self.num_affine == 3:
            scale, shift, gate = x.chunk(3, dim=-1)
        return scale, shift, gate

# ------------------------------------------------------------
# Multihead Self-Attention Block
# ------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch:int, num_heads:int=4, group:int=8):
        super().__init__()
        assert ch % num_heads == 0, f"channels {ch} must be divisible by heads {num_heads}"
        assert ch % group == 0, f"channels {ch} must be divisible by group {group}"

        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(group, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        x_norm = self.norm(x)
        qkv = self.qkv(x_norm) # (B, 3*C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1) # each is (B, C, H, W)

        # (B, C, H, W) -> (B, num_heads, head_dim, N) -> (B, num_heads, N, head_dim)
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.num_heads)

        # (B, h, N, d) @ (B, h, d, N) -> (B, h, N, N)
        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # (B, h, N, N) @ (B, h, N, d) -> (B, h, N, d)
        out = torch.einsum('bhij,bhjd->bhid', attn_probs, v)

        # merge
        # (B, h, N, d) -> (B, h, d, H, W) -> (B, C, H, W)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        out = self.proj(out)
        return x + out


# ------------------------------------------------------------
# Time-embedded Residual Block (+ Self-Attention)
# ------------------------------------------------------------
class TE_ResBlock(nn.Module):
    def __init__(self, ch_in:int, ch_out:int, temb_dim:int=128, group:int=8, 
                 num_att_heads:int=0, num_affine:int=2, act_layer=nn.SiLU, padding_mode='reflect'):
        super().__init__()
        assert ch_in % group == 0, f"ch_in {ch_in} must be divisible by group {group}"
        assert ch_out % group == 0, f"ch_out {ch_out} must be divisible by group {group}"
        assert num_affine in [2, 3], f"num_affine {num_affine} must be 2 or 3"

        self.norm1 = nn.GroupNorm(group, ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.norm2 = nn.GroupNorm(group, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.activation = act_layer()
        # place in the middle of the block
        self.film = FiLM_Layer_ada(ch_out, temb_dim, num_affine, act_layer) if temb_dim>0 else None

        if ch_in != ch_out:
            self.residual = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        else:
            self.residual = nn.Identity()

        self.attn = SelfAttention(ch_out, num_att_heads, group) if num_att_heads>0 else None

    def forward(self, x:torch.Tensor, temb:torch.Tensor=None) -> torch.Tensor:
        # norm(-film)-act-conv-norm-act-conv-residual(-attention)
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        h = self.norm2(h)
        if (self.film is not None) and (temb is not None):
            gate, scale, shift = self.film(temb)  # (B, ch_in)
            gate, scale, shift = gate[:, :, None, None], scale[:, :, None, None], shift[:, :, None, None]
            h = (scale + 1.) * h + shift # it ensures nearly identity mapping in early stages
            h = self.activation(h)
            h = self.conv2(h)
            h = h * gate # 
        else:
            h = self.activation(h)
            h = self.conv2(h)
        h = h + self.residual(x)

        if self.attn is not None:
            h = self.attn(h)

        return h

class ConvStem(nn.Module):
    def __init__(self, ch_in:int, ch_out:int, act_layer=nn.SiLU):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.activation = act_layer()
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation(x)
        h = self.conv2(x)
        return h

class hybrid_gui_attention(nn.Module):
    def __init__(self, ch_mask:int, ch_output:int=32, act_layer=nn.SiLU):
        super().__init__()
        self.mask_attention = nn.Sequential(
            nn.Linear(ch_mask, ch_output),
            act_layer(),
            nn.Linear(ch_output, ch_output),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_output, ch_output // 4, 1),
            act_layer(),
            nn.Conv2d(ch_output // 4, ch_output, 1),
            nn.Sigmoid()
        )

        self.fusion_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        mask_score = self.mask_attention(mask)[:, :, None, None]
        feature_score = self.channel_attention(x)
        alpha = torch.sigmoid(self.fusion_weight)
        combined_score = alpha * mask_score + (1 - alpha) * feature_score
        return x * combined_score
        

# ------------------------------------------------------------
# U-Net for Denoising Diffusion
# ------------------------------------------------------------
class denosing_unet(nn.Module):
    def __init__(self, ch_in_lst:int=1, ch_in_xt:int=1, 
                 num_gui:int=11, len_lulc=23, dim_lulc_embed=8, use_gate:bool=True,
                 ch_base:int=32, ch_mult:list=[1,2,4,8], ch_out:int=1,
                 act_layer=nn.SiLU, group:int=8, num_affine:int=2,
                 attn_levels:list=[3], num_att_heads:int=4, 
                 max_seq_len:int=15, temb_dim:int=128, max_period:float=10000, mlp_ratio:int=2,
                 residual_limit:float=40./180.):
        super().__init__()

        if ch_in_xt > 0:
            ## active the xt branch
            # time embedding
            self.xt_active = True
            self.timestep_embedder = TimestepEmbedder(max_seq_len, temb_dim, max_period, mlp_ratio)
            self.head_in_xt = ConvStem(ch_in=ch_in_xt, ch_out=ch_base, act_layer=act_layer)
            self.encoders_xt = nn.ModuleList()
            self.downs_xt = nn.ModuleList()
        else:
            self.xt_active = False

        ## geo branch
        self.head_in_lst = ConvStem(ch_in=ch_in_lst, ch_out=ch_base, act_layer=act_layer)
        self.lulc_embed_layer = nn.Embedding(len_lulc, dim_lulc_embed)
        self.dim_lulc_embed = dim_lulc_embed
        ch_gui_all = num_gui + dim_lulc_embed - 1
        self.head_in_gui = ConvStem(ch_in=ch_gui_all, ch_out=ch_base, act_layer=act_layer)
        
        if use_gate:
            self.gate_attention = hybrid_gui_attention(ch_mask=ch_gui_all, ch_output=ch_base, act_layer=act_layer)
        else:
            self.gate_attention = None
        
        self.encoders_geo = nn.ModuleList()
        self.downs_geo = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.head_out = nn.Conv2d(ch_base, ch_out, kernel_size=1)

        chs = [ch_base*i for i in ch_mult]
        for i in range(len(chs)-1):
            num_heads_i = num_att_heads if (i in attn_levels) else 0
            self.encoders_geo.append(TE_ResBlock(chs[i], chs[i+1], 0, group, num_heads_i, num_affine, act_layer))
            self.downs_geo.append(nn.Conv2d(chs[i+1], chs[i+1], kernel_size=3, stride=2, padding=1))
            if self.xt_active:
                self.encoders_xt.append(TE_ResBlock(chs[i], chs[i+1], temb_dim, group, num_heads_i, num_affine, act_layer))
                self.downs_xt.append(nn.Conv2d(chs[i+1], chs[i+1], kernel_size=3, stride=2, padding=1))
        
        num_heads_bridge = num_att_heads if (len(chs)-1 in attn_levels) else 0
        self.bridge = TE_ResBlock(chs[-1], chs[-1], temb_dim, group, num_heads_bridge, num_affine, act_layer)
        # self.bridge_geo = TE_ResBlock(chs[-1], chs[-1], 0, group, num_heads_bridge)

        for i in range(len(chs)-2, -1, -1):
            num_heads_i = num_att_heads if (i in attn_levels) else 0
            self.ups.append(nn.Sequential(
                nn.Conv2d(chs[i+1], chs[i+1]*4, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2)))
            self.decoders.append(TE_ResBlock(chs[i+1]*2, chs[i], temb_dim, group, num_heads_i, num_affine, act_layer))
        
        # physical constraint layer1: limit the output range
        self.residual_limit = residual_limit

    def forward(self, lst_lr_itp, ref, ndxi, dem, lulc, mask, xt=None, t=None):
        lulc_embed = self.lulc_embed_layer(lulc.squeeze(1)).permute(0, 3, 1, 2)  # (B, H, W, dim_lulc_embed) -> (B, dim_lulc_embed, H, W)
        extended_mask = torch.cat([mask, mask[:,-1:] * torch.ones(self.dim_lulc_embed-1, dtype=mask.dtype, device=mask.device)], dim=1) # (B, ch_gui_all)
        gui = torch.cat([ref, ndxi, dem, lulc_embed], dim=1)  # (B, ch_gui_all, H, W)
        gui = gui * extended_mask[:, :, None, None] # randomly drop channels according to mask
        
        x_lst = self.head_in_lst(lst_lr_itp)  # (B, ch_base, H, W)
        x_gui = self.head_in_gui(gui)  # (B, ch_base, H, W)
        if self.gate_attention is not None:
            x_gui = self.gate_attention(x_gui, extended_mask)
        x_geo = x_lst + x_gui
        
        if self.xt_active:
            temb = self.timestep_embedder(t) # (B, temb_dim)
            xt = self.head_in_xt(xt)
        else:
            temb = None

        enc_outs = []
        for i in range(len(self.encoders_geo)):
            x_geo = self.encoders_geo[i](x_geo, None)
            if self.xt_active:
                xt = self.encoders_xt[i](xt, temb)
                enc_outs.append(xt + x_geo)
                xt = self.downs_xt[i](xt)
            else:
                enc_outs.append(x_geo)
            x_geo = self.downs_geo[i](x_geo)
        
        if self.xt_active:
            x = self.bridge(xt + x_geo, temb)
        else:
            x = self.bridge(x_geo, temb)

        for decoder, up, enc_out in zip(self.decoders, self.ups, enc_outs[::-1]):
            x = up(x)
            x = torch.cat([x, enc_out], dim=1)
            x = decoder(x, temb)
        
        x = self.head_out(x)
        if self.residual_limit > 0:
            x = torch.clamp(x, -self.residual_limit, self.residual_limit) # limit the value range of dLST
        x = x + lst_lr_itp # LR LST + LST residual

        return x

if __name__ == "__main__":
    # test the model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, h, w = 4, 160, 160
    n_steps = 1000
    ch_in_lr, ch_in_gui, ch_in_xt = 1, 5, 0
    model = geo_unet(ch_in_gui=ch_in_gui, ch_in_lr=ch_in_lr, ch_in_xt=ch_in_xt, ch_out=1).to(device)
    x_gui = torch.randn(batch_size, ch_in_gui, h, w).to(device)
    x_lr = torch.randn(batch_size, ch_in_lr, h, w).to(device)
    xt = torch.randn(batch_size, ch_in_xt, h, w).to(device)
    t = torch.randint(0, n_steps, (batch_size,), dtype=torch.long).to(device)
    out = model(x_lr, x_gui, xt, t)
    print(out.shape)
    '''
    batch_size = 4
    lst_lr_itp = torch.randn(batch_size, 1, 160, 160).cuda()
    dem = torch.randn(batch_size, 1, 160, 160).cuda()
    ref = torch.randn(batch_size, 6, 160, 160).cuda()
    ndxi = torch.randn(batch_size, 3, 160, 160).cuda()
    lulc = torch.randint(0, 23, (batch_size, 1, 160, 160)).cuda()
    mask = torch.ones(batch_size, 11).cuda()
    # xt
    xt = torch.randn(batch_size, 1, 160, 160).cuda()
    n_steps = 15
    t = torch.randint(0, n_steps, (batch_size,), dtype=torch.long).cuda()
    # --------------
    model = denosing_unet(ch_in_lst=1, ch_in_xt=1, num_gui=11, len_lulc=23, dim_lulc_embed=8, use_gate=True,
                          max_seq_len=n_steps).cuda()
    output = model(lst_lr_itp, ref, ndxi, dem, lulc, mask, xt=xt, t=t)
    # model = denosing_unet(ch_in_lst=1, ch_in_xt=0, num_gui=11, len_lulc=23, dim_lulc_embed=8, use_gate=True).cuda()
    # output = model(lst_lr, gui_wo_lulc, lulc, mask)
    print(output.shape)