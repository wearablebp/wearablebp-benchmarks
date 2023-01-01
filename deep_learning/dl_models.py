from torch import nn
import torch.nn.functional as F
import torch

class jeong21(nn.Module):
    def __init__(self):
        super(jeong21, self).__init__()
        self.c1 = nn.Conv1d(in_channels=1, out_channels=56, kernel_size=10)
        self.b1 = nn.BatchNorm1d(num_features=56)
        self.d1 = nn.Dropout(p=0.5)
        self.l1 = nn.LSTM(input_size=56, hidden_size=28, num_layers=1, bidirectional=True, batch_first=True)
        self.l2 = nn.LSTM(input_size=56, hidden_size=28, num_layers=2, bidirectional=False, batch_first=True)
        self.a1 = nn.AvgPool1d(kernel_size=241, stride=1, padding=0)
        self.d2 = nn.Dropout(p=0.5)
        self.fc1_s = nn.Linear(in_features=28, out_features=28)
        self.fc2_s = nn.Linear(in_features=28, out_features=16)
        self.fc3_s = nn.Linear(in_features=16, out_features=1)
        self.fc1_d = nn.Linear(in_features=28, out_features=28)
        self.fc2_d = nn.Linear(in_features=28, out_features=16)
        self.fc3_d = nn.Linear(in_features=16, out_features=1)
        
    def forward(self, x):
#         print(x.shape)
        x = (x[:, 0, :] - x[:, 1, :]).unsqueeze(1)
        x = F.relu(self.c1(x))
        x = self.b1(x)
        x = self.d1(x).permute(0,2,1)
        x, (hn, cn) = self.l1(x)
        x, (hn, cn) = self.l2(x, (hn, cn))
        x = x.permute(0,2,1)
        x = self.a1(x)
        x = x.permute(0, 2, 1)
        x = self.d2(x)

        s = F.relu(self.fc1_s(x))
        s = F.relu(self.fc2_s(s))
        s = self.fc3_s(s)
        d = F.relu(self.fc1_d(x))
        d = F.relu(self.fc2_d(d))
        d = self.fc3_d(d)
        return torch.cat([s, d], axis=1)
    
    
# MLP-BP

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, causal = False, act = nn.Identity(), init_eps = 1e-3):
        super().__init__()
        dim_out = dim // 2
        self.causal = causal

        self.norm = nn.LayerNorm(dim_out)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)

        self.act = act

        init_eps /= dim_seq
        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x, gate_res = None):
        device, n = x.device, x.shape[1]

        res, gate = x.chunk(2, dim=-1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            weight, bias = weight[:n, :n], bias[:n]
            mask = torch.ones(weight.shape[:2], device = device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.)

        gate = F.conv1d(gate, weight, bias)

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res


class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
        attn_dim = None,
        causal = False,
        act = nn.Identity()
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.attn = Attention(dim, dim_ff // 2, attn_dim, causal) if exists(attn_dim) else None

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        gate_res = self.attn(x) if exists(self.attn) else None

        x = self.proj_in(x)
        x = self.sgu(x, gate_res = gate_res)
        x = self.proj_out(x)
        return x

class gMLP(nn.Module):
    def __init__(
        self,
        *,
        num_tokens = None,
        dim,
        depth,
        seq_len,
        ff_mult = 4,
        attn_dim = None,
        prob_survival = 1.,
        causal = False,
        act = nn.Identity()
    ):
        super().__init__()
        dim_ff = dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        self.to_embed = nn.Embedding(num_tokens, dim) if exists(num_tokens) else nn.Identity()

        self.layers = nn.ModuleList([Residual(PreNorm(dim, gMLPBlock(dim = dim, dim_ff = dim_ff, seq_len = seq_len, attn_dim = attn_dim, causal = causal, act = act))) for i in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        ) if exists(num_tokens) else nn.Identity()

    def forward(self, x):
        x = self.to_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        out = nn.Sequential(*layers)(x)
        return self.to_logits(out)


class huang22(nn.Module):
    def __init__(
        self,
        *,
        # image_size,
        in_channel,
        num_patches,
        num_classes,
        dim,
        depth,
        ff_mult = 6,
        attn_dim = None,
        prob_survival = 1.
    ):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        # assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        # num_patches = (image_height // patch_height) * (image_width // patch_width)

        dim_ff = dim * ff_mult

        self.to_patch_embed = nn.Sequential(
            Rearrange('b c t -> b t c'),
            nn.Linear(in_channel, dim)
        )

        self.prob_survival = prob_survival

        self.layers = nn.ModuleList([Residual(PreNorm(dim, gMLPBlock(dim = dim, dim_ff = dim_ff, seq_len = num_patches, attn_dim = attn_dim))) for i in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(num_patches, 2 * num_patches, kernel_size=6, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv1d(2 * num_patches, 4 * num_patches, kernel_size=6, stride=4, padding=1),
            nn.ReLU(),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(4 * num_patches, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        x = self.to_logits(x)
        x = x.mean(dim=2)
        x = self.mlp_head(x)
        return x