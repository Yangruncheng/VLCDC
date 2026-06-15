from __future__ import absolute_import, print_function

import argparse
import collections
import os
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import DBSCAN
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

PROJECT_DIR = osp.abspath(osp.dirname(osp.dirname(__file__)))
sys.path.insert(0, PROJECT_DIR)

from clustercontrast import datasets
from clustercontrast.config import cfg
from clustercontrast.evaluators import Evaluator, extract_features_sg
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.models.make_model_clipreid import make_model_sg
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint

best_mAP = 0


class AdaptiveStripMask:
    """Adaptive strip erasing used by the SG training pipeline."""

    def __init__(
        self,
        probability=0.7,
        semantic_aware=True,
        gradient_masking=True,
        v_num_bands=(2, 4),
        v_width_range=(15, 40),
        h_num_bands=(1, 2),
        h_width_range=(40, 60),
        ratio=0.7,
    ):
        self.probability = probability
        self.semantic_aware = semantic_aware
        self.gradient_masking = gradient_masking
        self.v_num_bands = v_num_bands
        self.v_width_range = v_width_range
        self.h_num_bands = h_num_bands
        self.h_width_range = h_width_range
        self.ratio = ratio

    def __call__(self, img):
        if random.random() > self.probability:
            return img

        image = np.array(img)
        h, w = image.shape[:2]
        saliency = self._saliency(image) if self.semantic_aware else None
        strips = self._make_strips(saliency, h, w)

        mask = np.ones((h, w), dtype=np.float32)
        for direction, start, end in strips:
            if direction == "v":
                mask[:, start:end] = 0
            else:
                mask[start:end, :] = 0

        if self.gradient_masking:
            mask = self._soften_edges(mask, strips)

        fill_color = (
            np.random.randint(0, 255, 3)
            if random.random() > 0.5
            else np.mean(image, axis=(0, 1)).astype(int)
        )
        mask = np.repeat(mask[:, :, None], 3, axis=2)
        erased = image.astype(np.float32) * mask + fill_color * (1 - mask)
        return Image.fromarray(np.clip(erased, 0, 255).astype(np.uint8))

    @staticmethod
    def _saliency(image):
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        try:
            import cv2

            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        except ImportError:
            grad_y, grad_x = np.gradient(gray)
            laplacian = np.abs(np.gradient(grad_x)[1] + np.gradient(grad_y)[0])
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)

        def normalize(x):
            return x / (x.max() + 1e-12)

        return normalize(gradient) * 0.4 + normalize(laplacian) * 0.3 + normalize(spectrum) * 0.4

    def _make_strips(self, saliency, h, w):
        strips = []
        v_count = random.randint(*self.v_num_bands)
        h_count = random.randint(*self.h_num_bands)

        for _ in range(v_count):
            if random.random() <= self.ratio:
                width = random.randint(*self.v_width_range)
                start = self._weighted_start(saliency, axis=0, size=w, width=width)
                strips.append(("v", start, min(start + width, w)))

        for _ in range(h_count):
            if random.random() <= self.ratio:
                width = random.randint(*self.h_width_range)
                start = self._weighted_start(saliency, axis=1, size=h, width=width)
                strips.append(("h", start, min(start + width, h)))

        return strips

    @staticmethod
    def _weighted_start(saliency, axis, size, width):
        max_start = max(1, size - width)
        if saliency is None or random.random() >= 0.7:
            return random.randint(0, max_start)

        weights = np.mean(saliency, axis=axis)
        weights = weights[:max_start]
        weights_sum = weights.sum()
        if weights_sum <= 0:
            return random.randint(0, max_start)
        return int(np.random.choice(max_start, p=weights / weights_sum))

    @staticmethod
    def _soften_edges(mask, strips):
        for direction, start, end in strips:
            center = (start + end) / 2
            radius = max(1, (end - start) / 2)
            for pos in range(start, end):
                strength = np.exp(-((abs(pos - center) / radius) ** 2) / 0.5)
                if direction == "v":
                    mask[:, pos] *= 1 - strength
                else:
                    mask[pos, :] *= 1 - strength
        return mask


def build_parser():
    parser = argparse.ArgumentParser("SG self-paced contrastive ReID training")
    parser.add_argument(
        "--config-file",
        "--config_file",
        default=osp.join(PROJECT_DIR, "clustercontrast", "configs", "person", "vit_clipreid.yml"),
        help="path to the SG/CLIP-ReID config file",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    parser.add_argument("-d", "--dataset", default="market1501", choices=datasets.names())
    parser.add_argument("--data-dir", default="", metavar="PATH")
    parser.add_argument("--logs-dir", default=osp.join(osp.dirname(__file__), "logs"), metavar="PATH")
    parser.add_argument("-b", "--batch-size", type=int, default=256)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-instances", type=int, default=8)

    parser.add_argument("--eps", type=float, default=0.60)
    parser.add_argument("--k1", type=int, default=30)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.2)

    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--step-size", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--eval-step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--self-norm", action="store_true")
    parser.add_argument("--use-hard", action="store_true")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def get_data(name, data_dir):
    return datasets.create(name, data_dir)


def normalizer(args):
    if args.self_norm:
        return T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_train_loader(args, dataset, trainset=None):
    transform = T.Compose(
        [
            T.Resize((args.height, args.width), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            AdaptiveStripMask(),
            T.Pad(10),
            T.RandomCrop((args.height, args.width)),
            T.ToTensor(),
            normalizer(args),
            T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
        ]
    )

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    sampler = RandomMultipleGallerySampler(train_set, args.num_instances) if args.num_instances > 0 else None
    loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir, transform=transform),
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
        shuffle=sampler is None,
        pin_memory=True,
        drop_last=True,
    )
    return IterLoader(loader, length=args.iters if args.iters > 0 else None)


def get_test_loader(args, dataset, testset=None):
    transform = T.Compose(
        [
            T.Resize((args.height, args.width), interpolation=3),
            T.ToTensor(),
            normalizer(args),
        ]
    )
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    return DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=transform),
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )


def create_sg_model(config):
    model = make_model_sg(config, num_classes=0, camera_num=0, view_num=0).cuda()
    return nn.DataParallel(model)


def configure_optimizer(model, args):
    trainable_scopes = (
        "inversion_network",
        "visual",
        "bottleneck_proj",
        "local.attention",
        "local.norm",
    )
    param_groups = []
    for scope in trainable_scopes:
        params = [p for name, p in model.named_parameters() if scope in name and p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": args.lr})

    if not param_groups:
        raise RuntimeError("No trainable SG parameters were found. Check model freezing rules.")

    return torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for index, label in enumerate(labels):
        if label != -1:
            centers[label].append(features[index])
    return torch.stack([torch.stack(centers[key], dim=0).mean(0) for key in sorted(centers.keys())], dim=0)


def print_cluster_statistics(labels, epoch):
    valid_labels = sorted(label for label in set(labels) if label != -1)
    noise = int(np.sum(labels == -1))
    print(f"=== Epoch {epoch} clustering statistics ===")
    print(f"Total clusters: {len(valid_labels)}")
    print(f"Noise samples: {noise} ({noise / len(labels) * 100:.2f}%)")

    counts = sorted(((label, int(np.sum(labels == label))) for label in valid_labels), key=lambda x: x[1], reverse=True)
    if not counts:
        print("No valid clusters were produced.")
        return

    print("Top 5 largest clusters:")
    for rank, (label, size) in enumerate(counts[:5], start=1):
        print(f"  {rank}. Cluster {label}: {size} samples")

    print("Top 5 smallest clusters:")
    for rank, (label, size) in enumerate(reversed(counts[-5:]), start=1):
        print(f"  {rank}. Cluster {label}: {size} samples")

    sizes = [size for _, size in counts]
    print(f"Min/Max/Avg cluster size: {min(sizes)} / {max(sizes)} / {np.mean(sizes):.2f}")


def build_memory(model, cluster_features, local_features, local_features_1, args):
    memory = ClusterMemory(
        512,
        cluster_features.size(0),
        temp=args.temp,
        momentum=args.momentum,
        use_hard=args.use_hard,
    ).cuda()
    memory.features = F.normalize(cluster_features, dim=1).cuda()

    inversion_network = model.module.inversion_network
    text_center = inversion_network(F.normalize(local_features, dim=1).cuda())
    text_center_2 = inversion_network(F.normalize(local_features_1, dim=1).cuda())
    global_center = inversion_network(F.normalize(cluster_features, dim=1).cuda())
    text_center = model(text_center, feat=text_center_2, glo=global_center, text=True)
    memory.text_center = F.normalize(text_center, dim=1).cuda()
    return memory


def train(args, config):
    global best_mAP
    start_time = time.monotonic()
    os.makedirs(args.logs_dir, exist_ok=True)
    sys.stdout = Logger(osp.join(args.logs_dir, "log.txt"))

    print("==========\nArgs:{}\n==========".format(args))
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(args, dataset)

    print("==> Build SG model")
    model = create_sg_model(config)
    evaluator = Evaluator(model)
    optimizer = configure_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    trainer = ClusterContrastTrainer(model)

    for epoch in range(args.epochs):
        print("==> Create pseudo labels for unlabeled data")
        cluster_loader = get_test_loader(args, dataset, testset=sorted(dataset.train))

        with torch.no_grad():
            features, _, local, local_1 = extract_features_sg(model, cluster_loader, print_freq=50)
            ordered_train = sorted(dataset.train)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in ordered_train], 0)
            local_features = torch.cat([local[f].unsqueeze(0) for f, _, _ in ordered_train], 0)
            local_features_1 = torch.cat([local_1[f].unsqueeze(0) for f, _, _ in ordered_train], 0)

            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
            cluster = DBSCAN(eps=args.eps, min_samples=4, metric="precomputed", n_jobs=-1)
            pseudo_labels = cluster.fit_predict(rerank_dist)

        print_cluster_statistics(pseudo_labels, epoch)
        cluster_features = generate_cluster_features(pseudo_labels, features)
        cluster_features_local = generate_cluster_features(pseudo_labels, local_features)
        cluster_features_local_1 = generate_cluster_features(pseudo_labels, local_features_1)

        trainer.memory = build_memory(
            model,
            cluster_features,
            cluster_features_local,
            cluster_features_local_1,
            args,
        )

        pseudo_labeled_dataset = [
            (fname, label, cid)
            for (fname, _, cid), label in zip(sorted(dataset.train), pseudo_labels)
            if label != -1
        ]
        print(f"==> Epoch {epoch}: {cluster_features.size(0)} clusters, {len(pseudo_labeled_dataset)} samples")

        train_loader = get_train_loader(args, dataset, trainset=pseudo_labeled_dataset)
        train_loader.new_epoch()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.8f}")
        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or epoch == args.epochs - 1:
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint(
                {"state_dict": model.state_dict(), "epoch": epoch + 1, "best_mAP": best_mAP},
                is_best,
                fpath=osp.join(args.logs_dir, "checkpoint.pth.tar"),
            )
            print(f"\n * Finished epoch {epoch:3d}  model mAP: {mAP:5.1%}  best: {best_mAP:5.1%}{' *' if is_best else ''}\n")

        scheduler.step()

    print("==> Test with the best model")
    checkpoint = load_checkpoint(osp.join(args.logs_dir, "model_best.pth.tar"))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
    print("Total running time:", timedelta(seconds=time.monotonic() - start_time))


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(args.seed if args.seed is not None else cfg.SOLVER.SEED)
    train(args, cfg)


if __name__ == "__main__":
    main()
