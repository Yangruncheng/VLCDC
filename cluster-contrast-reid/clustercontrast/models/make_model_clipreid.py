import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from  .clip import tokenize
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import List, Tuple, Optional
import torch.nn.functional as F
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_attention(m):
    classname = m.__class__.__name__
    # 处理MultiheadAttention内部的线性层（q/k/v投影和输出投影）
    if classname.find('Linear') != -1:
        # 对注意力中的线性层用Xavier初始化
        nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    # 处理LayerNorm的仿射参数（若存在）
    elif classname.find('LayerNorm') != -1:
        # 检查是否有可学习的weight（即affine=True的情况）
        if hasattr(m, 'weight'):
            nn.init.constant_(m.weight, 1.0)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, 0.0)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection       #CLIP 模型中的文本投影矩阵，用于将归一化后的特征投影到指定的维度。
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # prompts：输入的文本提示，是一个形状为 [batch_size, n_ctx, embedding_dim] 的张量，其中 n_ctx 是上下文标记的数量，embedding_dim 是嵌入向量的维度。
        # tokenized_prompts：分词后的文本提示，是一个形状为 [batch_size, seq_length] 的张量，其中 seq_length 是文本序列的长度。

        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection #从归一化后的特征中提取每个序列的结束标记（EOT，End-of-Sequence Token）对应的特征向量，并通过文本投影矩阵将其投影到指定的维度。
        return x        #返回最终的文本特征向量，形状为 [batch_size, projection_dim]，其中 projection_dim 是文本投影矩阵的输出维度。

class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, resnet50,camera_num, view_num):
        super(build_transformer, self).__init__()
        # 从配置文件中获取模型名称、余弦层、颈部特征等信息
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        # 根据模型名称设置输入通道数
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        # 保存类别数量、相机数量、视角数量和 SIE 系数
        self.num_classes = num_classes
        #self.camera_num = camera_num
        #self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        # 定义分类器并初始化权重
        #print(f"Type of in_planes: {type(self.in_planes)}, Value: {self.in_planes}")
        #print(f"Type of num_classes: {type(self.num_classes)}, Value: {self.num_classes}")
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        # 定义批归一化层并初始化权重
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        # 计算图像的分辨率
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        # 加载 CLIP 模型到 CPU 并移动到 GPU
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.cuda()
        # 提取 CLIP 模型的视觉编码器
        self.image_encoder = clip_model.visual
        # 根据配置添加相机和视角嵌入
        # if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
        #     self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_CAMERA:
        #     self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_VIEW:
        #     self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        # 初始化提示学习器和文本编码器
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        #self.resnet = resnet50
        self.num_features = self.in_planes_proj


    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        # 如果只需要文本特征
        if get_text == True:
            prompts = self.prompt_learner(label)        #生成文本提示
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)       #对生成的提示进行编码，得到文本特征。
            #return text_features
            return text_features
        # 如果只需要图像特征
        if get_image == True:
            #return self.resnet(x)           #新增，使用正常的resnet提取图片特征
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':       #如果模型是 RN50，则返回 image_features_proj 的第一个元素。如果模型是 ViT-B-16，则返回 image_features_proj 的第一列。
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        # 根据模型名称处理图像特征
        if self.model_name == 'RN50':
            #如果模型是 RN50，则对 image_features_last 和 image_features 进行平均池化操作，将其转换为一维向量，并取 image_features_proj 的第一个元素作为 img_feature_proj。
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            #如果模型是 ViT-B-16，则根据相机标签和视角标签计算上下文嵌入（cv_embed），并将其作为额外输入传递给 self.image_encoder。
            # 然后取 image_features_last、image_features 和 image_features_proj 的第一列作为相应的特征。
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]
        # 通过批归一化层处理特征
        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:   #使用 self.classifier 和 self.classifier_proj 对归一化后的特征进行分类，得到分类得分 cls_score 和 cls_score_proj。
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            #返回分类得分列表、原始图像特征列表和投影后的图像特征
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")   将归一化后的特征 feat 和 feat_proj 按维度 1 拼接并返回。
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)    #将原始特征 img_feature 和 img_feature_proj 按维度 1 拼接并返回


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, resnet50,num_class):
    model = build_transformer(num_class,cfg,resnet50,0,0)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]   #clip._MODELS 是一个字典，它将骨干网络名称映射到对应的模型下载 URL。通过 backbone_name 作为键，可以获取到相应的下载 URL。
    model_path = clip._download(url)        #函数会根据给定的 URL 下载模型文件，并返回模型文件在本地的路径。

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()       #尝试以 JIT（Just-In-Time）格式加载模型文件，并将其映射到 CPU 上。.eval() 方法将模型设置为评估模式，这意味着模型在推理过程中不会进行梯度计算，例如关闭 Dropout 层等。
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):    #token_embedding：用于将文本标记转换为嵌入向量的函数
        super().__init__()
        ctx_init = "A photo of a X X X X person."

        ctx_dim = 512   #上下文向量的维度，这里设置为 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")       #将初始提示文本中的下划线替换为空格。
        n_ctx = 4       #上下文标记的数量，这里设置为 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda()      #使用 CLIP 的分词器对初始提示文本进行分词
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)  #将分词后的张量转换为嵌入向量，并指定数据类型。
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor  将分词后的张量保存为类的属性

        n_cls_ctx = 4       #每个类别上下文标记的数量，这里设置为 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)       #创建一个空的张量，用于存储每个类别的上下文向量。
        nn.init.normal_(cls_vectors, std=0.02)      #使用正态分布初始化类别上下文向量，标准差为 0.02。
        self.cls_ctx = nn.Parameter(cls_vectors)        #将类别上下文向量保存为可训练的参数。

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])       #注册缓冲区，用于存储不需要训练的张量。嵌入向量的前缀部分，用于拼接提示文本。
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])      #嵌入向量的后缀部分，用于拼接提示文本
        self.num_class = num_class  #保存类别数量和每个类别上下文标记的数量。
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]       #根据输入的标签选择对应的类别上下文向量。
        b = label.shape[0]      #获取输入标签的批量大小。
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1)        #将前缀和后缀向量扩展到与批量大小相同的维度。
            
        prompts = torch.cat(        #将前缀、类别上下文向量和后缀向量在维度 1 上拼接起来，形成最终的提示文本。
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts      #PromptLearner 类的主要作用是根据输入的标签生成对应的文本提示，用于后续的文本编码和图像 - 文本匹配任务。











#============================================================PCL================================================================================
def make_model_pcl(cfg, num_classes, camera_num, view_num):
    model = TransReID_pcl(num_classes, camera_num, view_num, cfg)
    return model

def make_model_sg(cfg, num_classes, camera_num, view_num):
    model = PromptSGModel(num_classes, camera_num, view_num, cfg)
    return model







class MultiModalInteractionModule(nn.Module):
    """多模态交互模块：实现文本引导的跨注意力 + 后续Transformer块"""

    def __init__(self, embed_dim: int, num_heads: int, num_transformer_layers: int = 2):
        super().__init__()
        # 跨注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 后续Transformer块
        # self.transformer_blocks = nn.ModuleList([
        #     nn.TransformerEncoderLayer(
        #         d_model=embed_dim,
        #         nhead=num_heads,
        #         dim_feedforward=embed_dim * 4,
        #         activation="gelu",
        #         batch_first=True
        #     ) for _ in range(num_transformer_layers)
        # ])

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        #self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, text_embedding: torch.Tensor, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embedding: 文本嵌入 [batch_size, 1, embed_dim]
            patch_embeddings: 补丁嵌入 [batch_size, num_patches, embed_dim]
        Returns:
            processed_features: 处理后的特征 [batch_size, num_patches, embed_dim]
        """
        # 跨注意力层
        attended_patches, _ = self.cross_attention(
            query=text_embedding,
            key=patch_embeddings,
            value=patch_embeddings
        )

        # 第一个残差连接和层归一化
        x = self.norm1(patch_embeddings + attended_patches)

        # 通过多层Transformer块处理
        # for block in self.transformer_blocks:
        #     x = block(x)

        # 第二个残差连接和层归一化（可选）
        #processed_features = self.norm2(x)
        processed_features = x
        return processed_features

#文本翻转网络
class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)
        self.apply(weights_init_kaiming)
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)





class LocalFeatureExtractor_2(nn.Module):
    def __init__(self, embed_dim, num_heads=4, total_k=40):
        super().__init__()
        self.total_k = total_k  # 总共选择的patch数量
        self.k1 = total_k // 2  # 第一个分支的patch数量（前一半）
        self.k2 = total_k - self.k1  # 第二个分支的patch数量（后一半）
        self.embed_dim = embed_dim

        # 共享的注意力机制
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(weights_init_attention)

        print(f"LocalFeatureExtractor配置: total_k={total_k}, k1={self.k1}, k2={self.k2}")

    def forward(self, global_feat, patch_embeddings):
        """
        Args:
            global_feat: [B, D] 全局特征
            patch_embeddings: [B, N, D] patch特征，N=128
        Returns:
            local_feat_1: [B, D] 最重要的一半patch的特征
            local_feat_2: [B, D] 次重要的一半patch的特征
        """
        batch_size, num_patches, feat_dim = patch_embeddings.shape

        # 计算全局特征与所有patch的余弦相似度
        similarity = F.cosine_similarity(
            global_feat.unsqueeze(1),  # [B, 1, D]
            patch_embeddings,  # [B, N, D]
            dim=-1
        )  # [B, N]

        # 一次性获取top total_k个patch
        topk_similarity, topk_indices = torch.topk(
            similarity, k=self.total_k, dim=-1, sorted=True
        )  # [B, total_k]

        # 将top total_k分成两部分
        topk_indices_1 = topk_indices[:, :self.k1+5]  # 最重要的k1个patch
        topk_indices_2 = topk_indices[:, self.k1-5:self.total_k]  # 次重要的k2个patch

        # 提取对应的patch特征
        batch_indices = torch.arange(batch_size, device=patch_embeddings.device).unsqueeze(1)

        selected_patches_1 = patch_embeddings[batch_indices, topk_indices_1]  # [B, k1, D]
        selected_patches_2 = patch_embeddings[batch_indices, topk_indices_2]  # [B, k2, D]

        # 对两个分支分别应用注意力机制
        with torch.cuda.amp.autocast(enabled=True):
            # 第一个分支（最重要的patch）
            attended_1, _ = self.attention(
                selected_patches_1, selected_patches_1, selected_patches_1
            )
            attended_1 = self.norm(selected_patches_1 + attended_1)

            # 第二个分支（次重要的patch）
            attended_2, _ = self.attention(
                selected_patches_2, selected_patches_2, selected_patches_2
            )
            attended_2 = self.norm(selected_patches_2 + attended_2)

        # 池化得到最终特征
        local_feat_1 = torch.mean(attended_1, dim=1)  # [B, D]
        local_feat_2 = torch.mean(attended_2, dim=1)  # [B, D]

        # 可选：返回相似度信息用于分析或可视化
        # similarity_info = {
        #     'topk_similarity': topk_similarity,
        #     'topk_indices_1': topk_indices_1,
        #     'topk_indices_2': topk_indices_2
        # }

        return local_feat_1, local_feat_2

class LocalFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=40, k2=30):
        super().__init__()
        self.k = k
        self.k2 = k2
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(weights_init_attention)

    def forward(self, global_feat, patch_embeddings):
        # 只计算一次相似度
        similarity = F.cosine_similarity(global_feat.unsqueeze(1), patch_embeddings, dim=-1)

        # 一次性获取两个topk
        _, top_k_indices = torch.topk(similarity, k=self.k, dim=-1)
        _, top_k_indices_2 = torch.topk(similarity, k=self.k2, dim=-1)

        batch_size = patch_embeddings.size(0)

        # 提取Top-K patches
        batch_indices = torch.arange(batch_size, device=patch_embeddings.device).unsqueeze(1)
        selected_patches = patch_embeddings[batch_indices, top_k_indices]
        selected_patches_2 = patch_embeddings[batch_indices, top_k_indices_2]

        # 使用更高效的内存管理
        with torch.cuda.amp.autocast(enabled=True):  # 混合精度
            attended_patches, _ = self.attention(selected_patches, selected_patches, selected_patches)
            attended_patches = self.norm(selected_patches + attended_patches)

            attended_patches_2, _ = self.attention(selected_patches_2, selected_patches_2, selected_patches_2)
            attended_patches_2 = self.norm(selected_patches_2 + attended_patches_2)

        local_feat = torch.mean(attended_patches, dim=1)
        local_feat_2 = torch.mean(attended_patches_2, dim=1)

        return local_feat, local_feat_2
class PromptSGModel(nn.Module):
    """完整的PromptSG模型架构"""

    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME

        self.in_planes = 768
        self.in_planes_proj = 512
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.bottleneck_proj_local = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj_local.bias.requires_grad_(False)
        self.bottleneck_proj_local.apply(weights_init_kaiming)

        self.bottleneck_proj_local_1 = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj_local_1.bias.requires_grad_(False)
        self.bottleneck_proj_local_1.apply(weights_init_kaiming)

        # self.classifier = nn.Linear(self.in_planes_proj + self.in_planes, num_classes, bias=False)
        # self.classifier.apply(weights_init_classifier)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        # 加载预训练CLIP模型
        self.clip_model = load_clip_to_cpu_pcl(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)

        for _, v in self.clip_model.visual.conv1.named_parameters():
            v.requires_grad_(False)
        print('Freeze patch projection layer with shape {}'.format(self.clip_model.visual.conv1.weight.shape))

        num = 0
        for _, v in self.clip_model.transformer.named_parameters():
            v.requires_grad_(False)
            num+=1
        print('Freeze text_encoder, total: ',num)

        # 初始化文本反转网络
        self.inversion_network = IM2TEXT(
            embed_dim=512,
            middle_dim=512,
            output_dim=512,
            n_layer=3,
            dropout=0.1
        )

        # 初始化多模态交互模块
        # self.mim = MultiModalInteractionModule(
        #     embed_dim=512,
        #     num_heads=8,
        #     num_transformer_layers=2
        # )
        self.local = LocalFeatureExtractor(512,4,40,35)     #   embed_dim, num_heads,K1 , K2   patch16:      market1501 40 70    msmt17 40 50      patch32:  16  24
        #self.local = LocalFeatureExtractor_2(512, 4, 60)

        # 分类头
        #self.classifier = nn.Linear(self.patch_embed_dim, 0)  # 类别数在训练时设置

    def get_visual_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取视觉编码器的全局特征（CLS）和补丁特征"""
        # 前向传播CLIP视觉编码器，获取所有补丁+CLS的输出
        with torch.no_grad():
            x = self.clip_model.visual.conv1(images)  # shape: [batch_size, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [batch_size, width, grid*grid]
            x = x.permute(0, 2, 1)  # shape: [batch_size, grid*grid, width]
            x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) +
                           torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.clip_model.visual.ln_pre(x)

            # 提取所有transformer层的输出
            for i, resblock in enumerate(self.clip_model.visual.transformer.resblocks):
                x = resblock(x)
                # 提取最后一层的输出作为patch embeddings
                if i == len(self.clip_model.visual.transformer.resblocks) - 1:
                    patch_embeddings = x[:, 1:]  # 除去CLS token
                    global_feat = x[:, 0]  # CLS token作为全局特征

        return global_feat, patch_embeddings

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """使用CLIP文本编码器处理文本"""
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection

        return x

    def generate_prompt(self, image_features: torch.Tensor) -> torch.Tensor:
        """生成个性化提示"""
        # 获取伪令牌（修改点：适配IM2TEXT的输出）
        batch_size = image_features.shape[0]
        #pseudo_token_logits = self.inversion_network(image_features)


        # 构建组合提示："A photo of a [pseudo_token] person"
        template = clip.tokenize(["A photo of a person"]*batch_size, context_length=77).to(image_features.device)

        return template


    def forward(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None,feat = None,glo = None,text = None, return_inverted_text=False):

        if text:
            with torch.no_grad():  # 关闭梯度计算
                text = clip.tokenize(["A photo of person"], context_length=77).cuda()
                text = text.view(1, -1)
                text = text.repeat(images.size(0), 1)

                # text_1 = clip.tokenize(["A photo of person with and"], context_length=77).cuda()
                # text_1 = text_1.view(1, -1)
                # text_1 = text_1.repeat(images.size(0), 1)

            # text_2 = clip.tokenize(["A photo of person with"], context_length=77).cuda()
            # text_2 = text_2.view(1, -1)
            # text_2 = text_2.repeat(images.size(0), 1)

            #text_features = self.clip_model.encode_text_img_3(text_1, images,feat,glo)     # A photo of glo person with top-k1 and top-k2       feat是40
            text_features = self.clip_model.encode_text_img(text,glo)
            #text_features = self.clip_model.encode_text_img_2(text_1, images, feat, glo)
            #text_features = self.clip_model.encode_text_img_4(text_1, images, feat, glo)        # A photo of top-k1 person with top-k2
            return F.normalize(text_features)

        with torch.cuda.amp.autocast():
            # 提取图像特征（全局和补丁）
            _, image_features, image_features_proj, = self.clip_model.visual(images)
            #global_feat, patch_embeddings = self.get_visual_features(images)
            global_feat = image_features_proj[:,0]
            patch_embeddings = image_features_proj[:,1:]            #128个补丁

            # 及时释放不需要的变量
            del image_features, image_features_proj

            # # 添加的相似度计算代码
            # k = 40
            # similarity_scores = torch.cosine_similarity(
            #     global_feat.unsqueeze(1),  # [B, 1, D]
            #     patch_embeddings,  # [B, 128, D]
            #     dim=-1  # [B, 128]
            # )
            #
            # # 选择Top-K最相似的patches
            # top_k_scores, top_k_indices = torch.topk(similarity_scores, k=k, dim=-1)
            # # top_k_scores: [B, K] - Top-K相似度分数
            # # top_k_indices: [B, K] - Top-K patch索引
            #
            # # 提取Top-K patch特征
            # batch_size = patch_embeddings.size(0)
            # batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
            # selected_patches = patch_embeddings[batch_indices, top_k_indices]  # [B, K, D]
            #
            # # 5加权求和
            # weights = F.softmax(top_k_scores, dim=-1)  # [B, K] - 归一化权重
            # local_feat = torch.sum(selected_patches * weights.unsqueeze(-1), dim=1)  # [B, D]

            local_feat,local_feat_1 = self.local(global_feat,patch_embeddings)


            # # 生成提示
            # prompt_tokens = self.generate_prompt(global_feat)
            #
            # # 编码文本提示
            # text_features = self.encode_text(prompt_tokens)
            #
            # # 多模态交互
            # processed_features = self.mim(
            #     text_embedding=text_features.unsqueeze(1),  # [batch_size, 1, embed_dim]
            #     patch_embeddings=global_feat.unsqueeze(1)  # [batch_size, num_patches, embed_dim]
            # )
            # image_features = image_features[:,0]
            # image_features = self.bottleneck(image_features)
            # # 特征融合与分类
            # pooled_features = processed_features.mean(dim=1)  # 平均池化
            #
            # #pooled_features = self.bottleneck_proj(pooled_features)         #bn
            #
            # out_feat = torch.cat([image_features, pooled_features], dim=1)
            # out_feat = torch.nn.functional.normalize(out_feat)

            #logits = self.classifier(pooled_features)

            # 归一化特征
            f_local_2 = F.normalize(self.bottleneck_proj_local_1(local_feat_1))
            f_local = F.normalize(self.bottleneck_proj_local(local_feat))
            f_out = F.normalize(self.bottleneck_proj(global_feat))

            # *** 关键修改：如果需要翻转后的文本特征，一次性计算 ***
            if return_inverted_text:
                # 在同一个forward中完成翻转和文本生成
                inverted_local_2 = self.inversion_network(f_local_2)
                inverted_local = self.inversion_network(f_local)
                inverted_out = self.inversion_network(f_out)

                # 调用文本分支
                text_output = self.forward(images=inverted_local_2,
                                           feat=inverted_local,
                                           glo=inverted_out,
                                           text=True)
                return f_local_2, f_local, f_out, text_output

            return f_local_2, f_local, f_out

            return F.normalize(self.bottleneck_proj_local_1(local_feat_1)),F.normalize(self.bottleneck_proj_local(local_feat)),F.normalize(self.bottleneck_proj(global_feat))
            return {
                "global_features": global_feat,
                "text_features": text_features,
                "pooled_features": pooled_features,
                #"logits": logits
            }


class TransReID_pcl(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(TransReID_pcl, self).__init__()
        self.model_name = cfg.MODEL.NAME

        self.in_planes = 768
        self.in_planes_proj = 512
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        #self.classifier = nn.Linear(self.in_planes_proj + self.in_planes, num_classes, bias=False)
        #self.classifier.apply(weights_init_classifier)


        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu_pcl(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.encode_text_img = clip_model.encode_text_img


        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.image_encoder.conv1.named_parameters():
            v.requires_grad_(False)
        print('Freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

    def forward(self, x, cam_label=None, view_label=None):
        if cam_label != None and view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif cam_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            cv_embed = None
        _, image_features, image_features_proj, = self.image_encoder(x, cv_embed)
        img_feature = image_features[:, 0]
        img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)             #768
        feat_proj = self.bottleneck_proj(img_feature_proj)      #512

        out_feat = torch.cat([feat, feat_proj], dim=1)
        out_feat = torch.nn.functional.normalize(out_feat)

        # token_features = self.imag2text(feat_proj)           #文本翻转
        # text_features = self.get_text_features(token_features)
        # combine_feature = torch.cat(([out_feat,text_features]),dim=1)
        # combine_feature = torch.nn.functional.normalize(combine_feature)
        # return combine_feature

        return  torch.nn.functional.normalize(feat_proj)               # 512
        if self.training:
            #logit = self.classifier(out_feat)
            return out_feat#, logit
        else:
            return out_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if not self.training and 'classifier' in i:
                continue  # ignore classifier weights in evaluation
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))




#import clip.clip as clippp

def convert_tinyclip_to_clip(state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():

        # 去掉 DDP 的 module
        k = k.replace(".module", "")

        # image encoder → visual
        if k.startswith("_image_encoder."):
            k = k.replace("_image_encoder.", "")

        # text encoder → 直接映射
        elif k.startswith("_text_encoder."):
            k = k.replace("_text_encoder.", "")

        # logit scale
        elif k.startswith("_logit_scale."):
            k = k.replace("_logit_scale.", "")

        new_state_dict[k] = v

    return new_state_dict

def load_clip_to_cpu_pcl(backbone_name, h_resolution, w_resolution, vision_stride_size):



    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    if backbone_name == "TinyCLIP-ViT-39M-16":
        state_dict["state_dict"] = convert_tinyclip_to_clip(state_dict["state_dict"])

    #model = clip.build_model_tiny(state_dict["state_dict"], h_resolution, w_resolution, vision_stride_size)

    return model
