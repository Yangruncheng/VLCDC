#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss
from scipy.sparse import csr_matrix, lil_matrix
import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]           #前向搜索：找到样本 i 的前 k1+1 个最近邻（包括自身）
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]      #反向验证：检查这些邻居的前 k1+1 个最近邻是否包含样本 i
    fi = np.where(backward_k_neigh_index==i)[0]         #保留那些将 i 视为其近邻的样本，形成可靠的互近邻集合
    return forward_k_neigh_index[fi]            #样本 i 的互近邻索引列表

#通过多阶段近邻扩展和权重计算，生成鲁棒的 Jaccard距离矩阵，用于衡量样本间相似性
def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    #k1：初始搜索的最近邻数量，默认为 20。  k2：用于二次扩展的参数，默认为 6。
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()    #使用 faiss 库获取可用的 GPU 数量。
    N = target_features.size(0)     #样本数量
    mat_type = np.float16 if use_float16 else np.float32
    search_option = 2

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU       利用Faiss库高效计算每个样本的 k1 个最近邻
        index = index_init_gpu(ngpus, target_features.size(-1))     #初始化 GPU 索引
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)       #进行搜索[15618 30]
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []      #nn_k1 和 nn_k1_half 分别存储每个样本的 k 互近邻和 k/2 互近邻。
    nn_k1_half = []
    for i in range(N):  #通过循环调用 k_reciprocal_neigh 函数计算每个样本的互近邻
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))       #为每个样本生成 k1 互近邻列表(筛选出高置信度的正样本，减少噪声影响)
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)            #计算相似度矩阵 V
    for i in range(N):      #对于每个样本 i：计算 k 互近邻扩展索引 k_reciprocal_expansion_index。计算样本 i 与扩展索引中的样本之间的余弦距离 dist。根据 use_float16 的值，将相似度值存储在 V 矩阵中。
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)     #将局部近邻信息编码为权重矩阵，增强相似性度量的鲁棒性
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:     #二次扩展部分
        V_qe = np.zeros_like(V, dtype=mat_type)     #V_qe 是扩展后的相似度矩阵，通过对 V 矩阵的前 k2 个最近邻进行平均得到
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)       #对每个样本的前 k2 个近邻的相似度取平均，平滑权重矩阵（减少噪声，增强全局一致性。）
        V = V_qe
        del V_qe

    del initial_rank
    #计算 Jaccard 距离
    invIndex = []       #invIndex 存储每个样本在 V 矩阵中非零元素的索引
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):      #对于每个样本 i：计算 temp_min，表示两个样本之间的最小相似度。根据 temp_min 计算 Jaccard 距离。
        temp_min = np.zeros((1, N), dtype=mat_type)     #N为样本数量
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]      #出 V 矩阵中第 i 行非零元素的索引，并将这些索引存储在 indNonZero 数组中
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]   #获取与 indNonZero 中每个索引对应的图像索引，并将这些索引存储在 indImages 列表中
        for j in range(len(indNonZero)):
            #遍历 indNonZero 中的每个索引 j，计算 V[i, indNonZero[j]] 和 V[indImages[j], indNonZero[j]] 的最小值，并将这个最小值累加到 temp_min 中对应位置的元素上
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])   ## 计算样本i与其他样本的最小权重交集
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)           #通过权重交集度量样本相似性，值越小表示越相似
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)   #将 Jaccard 距离矩阵中小于零的元素设置为零
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print(f"Jaccard computing time: {time.time() - end:.2f}s")

    return jaccard_dist


