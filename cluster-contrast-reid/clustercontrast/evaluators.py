from __future__ import absolute_import, print_function

import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .utils import to_torch
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking


def extract_features_sg(model, data_loader, print_freq=50):
    """Extract global and local SG features."""
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    features = OrderedDict()
    labels = OrderedDict()
    local_k1 = OrderedDict()
    local_k2 = OrderedDict()
    end = time.time()

    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs = to_torch(imgs).cuda()
            local_2, local_1, outputs = model(imgs)

            outputs = outputs.data.cpu()
            local_1 = local_1.data.cpu()
            local_2 = local_2.data.cpu()
            for fname, output, loc_1, loc_2, pid in zip(fnames, outputs, local_1, local_2, pids):
                features[fname] = output
                local_k1[fname] = loc_1
                local_k2[fname] = loc_2
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(
                    "Extract Features: [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Data {:.3f} ({:.3f})\t".format(
                        i + 1,
                        len(data_loader),
                        batch_time.val,
                        batch_time.avg,
                        data_time.val,
                        data_time.avg,
                    )
                )

    return features, labels, local_k1, local_k2


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values())).view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        return dist.expand(n, n) - 2 * torch.mm(x, x.t())

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0).view(len(query), -1)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0).view(len(gallery), -1)
    m, n = x.size(0), y.size(0)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(
        dim=1, keepdim=True
    ).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist, x.numpy(), y.numpy()


def evaluate_all(
    query_features,
    gallery_features,
    distmat,
    query=None,
    gallery=None,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    cmc_topk=(1, 5, 10),
    cmc_flag=False,
):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert query_ids is not None and gallery_ids is not None
        assert query_cams is not None and gallery_cams is not None

    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print("Mean AP: {:4.1%}".format(mAP))

    if not cmc_flag:
        return mAP

    cmc_scores = cmc(
        distmat,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=True,
    )
    print("CMC Scores:")
    for k in cmc_topk:
        print("  top-{:<4}{:12.1%}".format(k, cmc_scores[k - 1]))
    return cmc_scores, mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _, _, _ = extract_features_sg(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(
            query_features,
            gallery_features,
            distmat,
            query=query,
            gallery=gallery,
            cmc_flag=cmc_flag,
        )

        if not rerank:
            return results

        print("Applying person re-ranking ...")
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(
            query_features,
            gallery_features,
            distmat,
            query=query,
            gallery=gallery,
            cmc_flag=cmc_flag,
        )
