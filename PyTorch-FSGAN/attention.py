import torch
import torch.nn as nn
# from deformable_attention import DeformableAttention


class SelfAttention(nn.Module):
    """ Self-Attention Layer """
    def __init__(self, in_dim, activation="relu", k=8):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.actiivation = activation

        # define q,k,v with Conv2d replaced Linear
        # Since we did not notice any significant performance decrease when reducing the channel number of C to be C/k, where k = 1,2,4,8 after few training epochs on ImageNet. For memory efficiency, we choose k=8 in all our experiments.
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.key   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        input:
            x : input feature maps (B, C, W, H)
        return:
            out : self attention value + input feature
            attention : (B, N, N), N is W*H
        """
        B, C, W, H = x.size()
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)  # (B, C//8, W, H) -> (B, C', N) -> (B, N, C')
        proj_key   = self.key(x).view(B, -1, W * H)  # (B, C//8, W, H) -> (B, C', N)
        qk = torch.bmm(proj_query, proj_key)  # (B, N, C') x (B, C', N) = (B, N, N)
        attention  = self.softmax(qk)  # (B, N, N)
        proj_value = self.value(x).view(B, -1, W * H)  # (B, C, W, H) -> (B, C, N)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, N) x (B, N, N) -> (B, C, N)
        out = out.view(B, C, W, H)  # (B, C, N) -> (B, C, W, H)

        out = self.gamma * out + x
        return out, attention

# test...
# x = torch.ones(size=(32, 64, 20, 20))
# out, attention = SelfAttention(64, "relu")(x)
# print(out.shape, attention.shape)


class MultiHeadSelfAttention(nn.Module):
    """ Multi Head Self-Attention """
    def __init__(self, in_dim, num_heads=4 , k=2):
        super(MultiHeadSelfAttention, self).__init__()
        # first compress, then multi-head
        assert (in_dim // k) % num_heads == 0  # q, k
        assert in_dim % num_heads == 0  # v
        
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.key   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.num_heads = num_heads
        self.k = k
        # self.scale = (in_dim // num_heads) ** -0.5
        
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        input:
            x : input feature maps (B, C, W, H)
        return:
            out : self attention value + input feature
            attention : (B, h, N, N), N is W*H, h is mutli-head
        """
        B, C, W, H = x.size()
        h = self.num_heads
        k = self.k
        proj_query = self.query(x).reshape(B, h, C//k//h, W*H).permute(0, 1, 3, 2)  # (B, C//2, W, H) -> (B, h, C'//h, N) -> (B, h, N, C'//h)
        proj_key = self.key(x).reshape(B, h, C//k//h, W*H)  # (B, C//2, W, H) -> (B, h, C'//h, N)
        qk = torch.matmul(proj_query, proj_key)  # (B, h, N, C'//h) x (B, h, C'//h, N) = (B, h, N, N)
        attention = self.softmax(qk) # (B, h, N, N)
        proj_value = self.value(x).reshape(B, h, C//h, W*H)  # (B, C, W, H) -> (B, h, C//h, N)
        
        out = torch.matmul(proj_value, attention.permute(0, 1, 3, 2))  # (B, h, C//h, N) x (B, h, N, N) = (B, h, C//h, N)
        out = out.reshape(B, C, W, H)  # (B, h, C//h, N) -> (B, C, W, H)
        
        out = self.gamma * out + x
        return out, attention
        
# test...
# x = torch.ones(size=(32, 64, 20, 20))
# out, attention = MultiHeadSelfAttention(64)(x)
# print(out.shape, attention.shape)


class MultiHeadDeformableAttention(nn.Module):
    """ Multi Head Deformable Attention """
    def __init__(self, in_dim, num_heads=4, dropout=0.):
        super(MultiHeadDeformableAttention, self).__init__()
        
        assert in_dim % num_heads == 0
        self.dim = in_dim
        self.dim_head = in_dim // num_heads
        self.heads = num_heads
        self.dropout = dropout
        
        self.gamma = nn.Parameter(torch.tensor([0.]))
        
    def forward(self, x):
        """
        input:
            x : input feature maps (B, C, W, H)
        return:
            out : self attention value + input feature, (B, C, W, H)
            attention : (B, C, W, H)
        """
        attn = DeformableAttention(
            dim=self.dim,
            dim_head=self.dim_head,
            heads=self.heads,
            dropout=self.dropout,
            downsample_factor=4,
            offset_scale=4,
            offset_groups=None,
            offset_kernel_size=6
        ).cuda()  # weight type -> torch.cuda.FloatTensor
        attention = attn(x)
        out = self.gamma * attention + x
        return out, attention

# test...
# x = torch.ones(size=(32, 64, 20, 20))
# out, attention = MultiHeadDeformableAttention(64)(x)
# print(out.shape, attention.shape)
