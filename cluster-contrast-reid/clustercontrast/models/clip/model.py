from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        #print(spacial_dim)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)             # output_dim or embed_dim
        self.num_heads = num_heads

    def forward(self, x): 
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC  #32,2048,7,7 ->49, 32, 2048
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC  50,32,2048
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,                            #原来q是x
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        ) 

        return x.squeeze(0)                     #原来是x

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1) 
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution, embed_dim, heads, output_dim)
        self.num_features = output_dim      #新增，特征数量
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): 
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype) 
        x = stem(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x3 = self.layer3(x) 
        x4 = self.layer4(x3) 
        xproj = self.attnpool(x4) 

        return x3, x4, xproj 


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)   #一个二维卷积层，用于将输入图像分割成多个 patch 并进行特征提取。

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  #一个可学习的分类嵌入向量，用于在 Transformer 中表示整个图像的类别信息。
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution*w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)        #一个自定义的 Transformer 模块，用于对 patch 序列进行特征编码。

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))    #一个可学习的投影矩阵，用于将 Transformer 的输出投影到指定的输出维度。
        self.num_features = output_dim

    def forward(self, x: torch.Tensor, cv_emb = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]     #对输入图像进行卷积操作，将其分割成多个 patch 并提取特征。
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]      #将卷积输出的特征图展平成二维张量。
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]     交换特征维度，使得特征张量的形状变为 [batch_size, num_patches, width]。
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]在 patch 序列的开头添加分类嵌入向量
        if cv_emb != None:  #如果提供了额外的嵌入向量 cv_emb，则将其加到分类嵌入向量上。
            x[:,0] = x[:,0] + cv_emb
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND    交换特征维度，将输入特征的形状从 [batch_size, num_patches, width] 转换为 [num_patches, batch_size, width]，以适应 Transformer 的输入要求
        
        x11 = self.transformer.resblocks[:11](x)        #将输入特征通过 Transformer 的前 11 个残差块进行编码。
        x12 = self.transformer.resblocks[11](x11)       #将前 11 个残差块的输出通过第 12 个残差块进行编码。
        x11 = x11.permute(1, 0, 2)  # LND -> NLD        #将编码后的特征维度交换回 [batch_size, num_patches, width]
        x12 = x12.permute(1, 0, 2)  # LND -> NLD  

        x12 = self.ln_post(x12)         #对第 12 个残差块的输出进行层归一化。

        if self.proj is not None:       #如果投影矩阵 self.proj 存在，则将归一化后的输出投影到指定的输出维度。
            xproj = x12 @ self.proj   
    #Transformer 前 11 个残差块的输出、第 12 个残差块的输出以及投影后的输出
        return x11, x12, xproj


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,        #图像和文本特征的维度
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],      #视觉编码器的层数，可以是一个整数或一个包含四个整数的元组。
                 vision_width: int,     #视觉编码器的宽度。
                 vision_patch_size: int,
                 vision_stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,        #文本编码器中 Transformer 的宽度。
                 transformer_heads: int,
                 transformer_layers: int,
                 h_resolution: int, 
                 w_resolution: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):        #如果 vision_layers 是一个元组或列表，则使用 ModifiedResNet 作为视觉编码器。
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=h_resolution*w_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                h_resolution = h_resolution,
                w_resolution = w_resolution,
                patch_size = vision_patch_size,
                stride_size = vision_stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
            #文本编码器
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.end_id = self.vocab_size - 1
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)      #将输入的文本标记转换为嵌入向量。
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)        #最终的层归一化层。

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))      #将文本特征投影到与图像特征相同的维度。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))      #用于调整对比损失的尺度。

        self.initialize_parameters()        #初始化模型的参数


    def initialize_parameters(self):        #使用正态分布初始化模型的参数，不同的组件使用不同的标准差。
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):     #创建一个因果注意力掩码，用于在文本编码时防止未来信息的泄露。
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)    #使用 torch.empty 函数创建一个形状为 (self.context_length, self.context_length) 的空张量 mask。self.context_length 表示输入文本序列的最大长度。
        mask.fill_(float("-inf"))       #使用 fill_ 方法将 mask 张量中的所有元素填充为负无穷（-inf）。
        mask.triu_(1)  # zero out the lower diagonal使用 triu_ 方法（原地操作）将 mask 张量的下三角部分（不包括对角线）置为零。参数 1 表示从主对角线的上一行开始保留元素，即只保留上三角部分。这样，每个位置只能关注到该位置及其之前的元素，实现了因果注意力机制。
        return mask

    @property
    def dtype(self):        #dtype 属性返回视觉编码器第一个卷积层的权重数据类型。
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):      #encode_image 方法将输入的图像转换为特征向量。
        return self.visual(image.type(self.dtype))

    def encode_text(self, text): 
        x = self.token_embedding(text).type(self.dtype)  

        x = x + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype) 
        #这是一个索引操作，通过 torch.arange(x.shape[0]) 和 text.argmax(dim=-1) 这两个索引数组，从 x 中提取出每个样本对应的特征向量。
        #从 x 的每一行中选取由 text.argmax(dim=-1) 所指定列的元素，最终得到一个形状为 (batch_size, feature_dim) 的张量，其中 feature_dim 是 x 的最后一个维度大小。
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection #

        return x

    def encode_text_img(self, text, img_tokens):        # 将文本与图像 token 融合后编码，输出联合特征。
        b_size = img_tokens.size(0)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]         将文本 token 转换为嵌入向量，形状变为 (batch_size, seq_len, d_model)
        collect_ind = text == self.end_id               # text == self.end_id：生成布尔张量，标记 text 中等于结束符 end_id 的位置
        collect_ind = collect_ind.nonzero()[:, 1]       # 提取每个样本中 end_id 的列索引（即序列中的位置）
        img_tokens = img_tokens.view(b_size, 1, -1)     # 将图像 token 从 (batch_size, img_feature_dim) 调整为 (batch_size, 1, img_feature_dim)，以便插入文本序列。
        x = torch.cat([x[:, :collect_ind[0]-1], img_tokens, x[:, collect_ind[0]-1:-1]], dim=1)      # 填充特征
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.size(0)), collect_ind+1] @ self.text_projection            # 取插入图像 token 的位置（即 end_id 的原始位置 +1）的特征。然后进行投影
        return x

    def encode_text_img_2(self, text, img_tokens, img_tokens_1,glo):        # 将文本与图像 token 融合后编码，输出联合特征。
        b_size = img_tokens.size(0)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]         将文本 token 转换为嵌入向量，形状变为 (batch_size, seq_len, d_model)
        collect_ind = text == self.end_id               # text == self.end_id：生成布尔张量，标记 text 中等于结束符 end_id 的位置
        collect_ind = collect_ind.nonzero()[:, 1]       # 提取每个样本中 end_id 的列索引（即序列中的位置）
        img_tokens = img_tokens.view(b_size, 1, -1)     # 将图像 token 从 (batch_size, img_feature_dim) 调整为 (batch_size, 1, img_feature_dim)，以便插入文本序列。
        img_tokens_1 = img_tokens_1.view(b_size, 1, -1)
        glo = glo.view(b_size, 1, -1)
        x = torch.cat([x[:, :collect_ind[0]-1],glo, img_tokens,img_tokens_1, x[:, collect_ind[0]-1:-3]], dim=1)      # 填充特征
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.size(0)), collect_ind+1] @ self.text_projection            # 取插入图像 token 的位置（即 end_id 的原始位置 +1）的特征。然后进行投影
        return x


    def encode_text_img_3(self, text, img_tokens, img_tokens_1,glo):        # 将文本与图像 token 融合后编码，输出联合特征。
        b_size = img_tokens.size(0)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]         将文本 token 转换为嵌入向量，形状变为 (batch_size, seq_len, d_model)
        collect_ind = text == self.end_id               # text == self.end_id：生成布尔张量，标记 text 中等于结束符 end_id 的位置
        collect_ind = collect_ind.nonzero()[:, 1]       # 提取每个样本中 end_id 的列索引（即序列中的位置）
        img_tokens = img_tokens.view(b_size, 1, -1)     # 将图像 token 从 (batch_size, img_feature_dim) 调整为 (batch_size, 1, img_feature_dim)，以便插入文本序列。
        img_tokens_1 = img_tokens_1.view(b_size, 1, -1)
        glo = glo.view(b_size, 1, -1)
        x = torch.cat([x[:, :4], glo,x[:,4:6],img_tokens_1,x[:,6:7],img_tokens, x[:, collect_ind[0]:-3]], dim=1)      # 填充特征
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.size(0)), collect_ind+1] @ self.text_projection            # 取插入图像 token 的位置（即 end_id 的原始位置 +1）的特征。然后进行投影
        return x

    def encode_text_img_4(self, text, img_tokens, img_tokens_1,glo):        # 将文本与图像 token 融合后编码，输出联合特征。
        b_size = img_tokens.size(0)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]         将文本 token 转换为嵌入向量，形状变为 (batch_size, seq_len, d_model)
        collect_ind = text == self.end_id               # text == self.end_id：生成布尔张量，标记 text 中等于结束符 end_id 的位置
        collect_ind = collect_ind.nonzero()[:, 1]       # 提取每个样本中 end_id 的列索引（即序列中的位置）
        img_tokens = img_tokens.view(b_size, 1, -1)     # 将图像 token 从 (batch_size, img_feature_dim) 调整为 (batch_size, 1, img_feature_dim)，以便插入文本序列。
        img_tokens_1 = img_tokens_1.view(b_size, 1, -1)
        glo = glo.view(b_size, 1, -1)
        x = torch.cat([x[:, :4], img_tokens_1,x[:,4:6],glo, x[:, collect_ind[0]:-2]], dim=1)      # 填充特征
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.size(0)), collect_ind+1] @ self.text_projection            # 取插入图像 token 的位置（即 end_id 的原始位置 +1）的特征。然后进行投影
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features   对图像特征和文本特征进行归一化处理，使得每个特征向量的模长为 1。
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()         #每个图像与所有文本之间的相似度得分矩阵。
        logits_per_text = logit_scale * text_features @ image_features.t()      #每个文本与所有图像之间的相似度得分矩阵。

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, h_resolution: int, w_resolution: int, vision_stride_size: int):
    vit = "visual.proj" in state_dict

    if vit :
        vision_width = state_dict["visual.conv1.weight"].shape[0]       #通过 visual.conv1.weight 的通道数确定。
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])   #视觉编码器的层数
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else: #RN50
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]        #统计 visual.layer1 到 visual.layer4 中不同块的数量。
        vision_layers = tuple(counts)       #将 counts 转换为元组。
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0] #77 (77,512)       #上下文长度
    vocab_size = state_dict["token_embedding.weight"].shape[0]      #词汇表大小
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))    #层数

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size, vision_stride_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        h_resolution, w_resolution
    )
    if vit:     #调整词嵌入大小
        state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"], model.visual.positional_embedding, h_resolution, w_resolution)
    else: #RN50
        state_dict["visual.attnpool.positional_embedding"] = resize_pos_embed(state_dict["visual.attnpool.positional_embedding"], model.visual.attnpool.positional_embedding, h_resolution, w_resolution)
    
    
    for key in ["input_resolution", "context_length", "vocab_size"]:        #删除不必要的键
        if key in state_dict:
            del state_dict[key]
            
    convert_weights(model)      #调用 convert_weights 函数将模型参数转换为合适的数据类型（通常是 fp16）

    model.load_state_dict(state_dict)
    return model.eval()


def build_model_tiny(state_dict: dict, h_resolution: int, w_resolution: int, vision_stride_size: int) -> object:
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]  # 通过 visual.conv1.weight 的通道数确定。
        vision_layers = len([k for k in state_dict.keys() if
                             k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])  # 视觉编码器的层数
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

    # _text_encoder.module.
    # _image_encoder.module.

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]  # 77 (77,512)       #上下文长度
    vocab_size = state_dict["token_embedding.weight"].shape[0]  # 词汇表大小
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))  # 层数

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size, vision_stride_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        h_resolution, w_resolution
    )
    if vit:  # 调整词嵌入大小
        state_dict["visual.positional_embedding"] = resize_pos_embed_tiny(state_dict["visual.positional_embedding"],
                                                                     model.visual.positional_embedding, h_resolution,
                                                                     w_resolution)


    for key in ["input_resolution", "context_length", "vocab_size"]:  # 删除不必要的键
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)  # 调用 convert_weights 函数将模型参数转换为合适的数据类型（通常是 fp16）

    model.load_state_dict(state_dict)
    return model.eval()

import math
def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
      
    ntok_new = posemb_new.shape[0] #129,2048

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid))) #14
    print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2) 
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear') 
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)
    return posemb


def resize_pos_embed_tiny(posemb, posemb_new, hight, width):
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)

    ntok_new = posemb_new.shape[0]

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Position embedding resize to height:{} width: {}'.format(hight, width))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)

    # 🔥 关键修复
    dtype = posemb_grid.dtype
    posemb_grid = posemb_grid.float()
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.to(dtype)

    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)

    posemb = torch.cat([posemb_token, posemb_grid.squeeze(0)], dim=0)
    return posemb