from __future__ import absolute_import, print_function

import time

import torch.nn.functional as F

from .utils.meters import AverageMeter


class ClusterContrastTrainer(object):
    """Trainer for the SG cluster-contrast objective."""

    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i in range(train_iters):
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            images, labels, _, _ = self._parse_data(inputs)
            local_k2, local_k1, image_features, text_features = self._forward_with_text(images)

            _, loss_image, loss_text = self.memory(image_features, text_features, labels)
            loss = loss_image + 0.1 * loss_text if epoch > 9 else loss_image

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 20 == 0:
                align_loss = self._alignment_loss(image_features, labels)
                print(loss.item(), align_loss.item(), loss_image.item(), loss_text.item())

            if (i + 1) % print_freq == 0:
                print(
                    "Epoch: [{}][{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Data {:.3f} ({:.3f})\t"
                    "Loss {:.3f} ({:.3f})".format(
                        epoch,
                        i + 1,
                        len(data_loader),
                        batch_time.val,
                        batch_time.avg,
                        data_time.val,
                        data_time.avg,
                        losses.val,
                        losses.avg,
                    )
                )

    def _parse_data(self, inputs):
        images, paths, pids, _, indexes = inputs
        return images.cuda(), pids.cuda(), indexes.cuda(), paths

    def _forward_with_text(self, inputs):
        return self.encoder(inputs, return_inverted_text=True)

    def _alignment_loss(self, image_features, labels):
        logits = image_features.mm(self.memory.text_center.t()) / 0.05
        return F.cross_entropy(logits, labels)
