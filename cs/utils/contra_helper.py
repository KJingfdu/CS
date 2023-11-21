import time
from PIL import Image
import numpy as np
import math
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from .sep_helper import separation, gradient_descent, separation_func


class MocoContrastLoss(nn.Module):
    def __init__(self, nclass, neg_num=64, temperature=0.1,
                 memory_bank=True, pixel_update_freq=50,
                 memory_size=2000, feat_dim=256,
                 mode='N+A+F', strategy='random',
                 device='cpu', ignore_label=255, eval_feat=True):
        super(MocoContrastLoss, self).__init__()
        # mode can only be chosed from ['N', 'A', 'N+A', 'N+A+F']
        self.strategy = strategy if strategy in ['random', 'semi-hard', 'boundary-interior'] else 'boundary-interior'
        self.mode = mode if mode in ['N', 'A', 'N+A', 'N+A+F'] else 'N+A'
        self.temperature = temperature
        self.ignore_label = ignore_label
        self.nclass = nclass
        self.topk = 3
        self.max_views = [neg_num for _ in range(nclass)]
        if memory_bank:
            self.memory_bank = MoCoMemoryBank(nclass, memory_size, pixel_update_freq, feat_dim, device)
        else:
            self.memory_bank = None
        if eval_feat:
            self.eval_bank = EvalFeat()
            self.eval_bank_2 = EvalFeat()

    # 这个random采样还是在各个类别上随机选取点采样的,还不是最普通的random
    def _random_sampling(self, X, X_t, y_hat, y, unlabeled=True, gt_y=None):
        batch_size = X.shape[0]
        y_hat = y_hat.contiguous().view(batch_size, -1)
        sampling_num = 1200
        if gt_y is not None:
            gt_y = gt_y.contiguous().view(batch_size, -1)
        X = X.contiguous().view(X.shape[0], -1, X.shape[-1])
        X_t = X_t.contiguous().view(X_t.shape[0], -1, X_t.shape[-1])
        feat_dim = X.shape[-1]
        X_ = torch.empty((0, feat_dim)).cuda()
        X_t_ = torch.empty((0, feat_dim)).cuda()
        y_ = torch.empty(0).long().cuda()
        Y_ = torch.empty(0).cuda()
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            ignored_y_hat = (this_y_hat != 255).nonzero().squeeze()
            num_pixels = ignored_y_hat.shape[0]
            perm = torch.randperm(num_pixels)
            K = min(sampling_num, num_pixels)
            indices = ignored_y_hat[perm[:K]]
            X_ = torch.cat((X_, X[ii, indices, :]), dim=0)
            X_t_ = torch.cat((X_t_, X_t[ii, indices, :]), dim=0)
            if gt_y is not None:
                Y_ = torch.cat((Y_, gt_y[ii, indices]))
            y_ = torch.cat((y_, this_y_hat[indices]))
        if gt_y is not None and unlabeled:
            return X_, X_t_, y_, Y_
        return X_, X_t_, y_

    def _area_sampling(self, X, X_t, y_hat, y, unlabeled=True, gt_y=None):
        batch_size = X.shape[0]
        feat_dim = X.shape[-1]
        X_ = torch.empty((0, feat_dim)).cuda()
        X_t_ = torch.empty((0, feat_dim)).cuda()
        y_ = torch.empty(0).cuda()
        Y_ = torch.empty(0).cuda()
        for ii in range(batch_size):
            # this_y = y[ii]
            # 根据student model预测出的y进行mask的挑选,进而挑选hard-easy sample.
            this_y_s = y_hat[ii]
            # 图片中存在255的部分, 因为采样是根据预测结果进行的, 很可能会涉及255的部分
            ignore_mask = (this_y_s == 255)
            this_X = X[ii]
            this_X_t = X_t[ii]
            this_y_ids = this_y_s.unique()
            this_y_ids = [i for i in this_y_ids if not i == 255]
            edge_mask, interior_mask = edge_interior(this_y_s, ignore_mask)
            # ignore的部分不予采样
            num_edge = [int(mask.sum()) for mask in edge_mask]
            num_interior = [int(mask.sum()) for mask in interior_mask]
            n_view = self.max_views[0]
            for id in range(len(num_edge)):
                edge = num_edge[id]
                interior = num_interior[id]
                if edge >= n_view / 2 and interior >= n_view / 2:
                    num_edge[id] = n_view // 2
                    num_interior[id] = n_view - num_edge[id]
                elif edge + interior >= n_view and interior < n_view / 2:
                    num_interior[id] = interior
                    num_edge[id] = n_view - num_interior[id]
                elif edge + interior >= n_view and edge < n_view / 2:
                    num_edge[id] = edge
                    num_interior[id] = n_view - num_edge[id]
            for id, value in enumerate(this_y_ids):
                num_all = edge_mask[id].int().sum()
                if num_all <= 0:
                    continue
                num_keep = num_edge[id]
                perm = torch.randperm(num_all)
                random_id = perm[:num_keep]
                X_t_ = torch.cat((X_t_, this_X_t[edge_mask[id]][random_id, :]), dim=0)
                X_ = torch.cat((X_, this_X[edge_mask[id]][random_id, :]), dim=0)
                y_ = torch.cat((y_, this_y_s[edge_mask[id]][random_id]))
                if gt_y is not None and unlabeled:
                    this_y = gt_y[ii]
                    Y_ = torch.cat((Y_, this_y[edge_mask[id]][random_id]))

                num_all = interior_mask[id].int().sum()
                if num_all <= 0:
                    continue
                num_keep = num_interior[id]
                perm = torch.randperm(num_all)
                random_id = perm[:num_keep]
                X_t_ = torch.cat((X_t_, this_X_t[interior_mask[id]][random_id, :]), dim=0)
                X_ = torch.cat((X_, this_X[interior_mask[id]][random_id, :]), dim=0)
                y_ = torch.cat((y_, this_y_s[interior_mask[id]][random_id]))
                if gt_y is not None and unlabeled:
                    this_y = gt_y[ii]
                    Y_ = torch.cat((Y_, this_y[interior_mask[id]][random_id]))
        if gt_y is not None and unlabeled:
            return X_, X_t_, y_, Y_
        return X_, X_t_, y_

    def _semi_sampling(self, X, X_t, y_hat, y, unlabeled=True, gt_y=None):
        batch_size = X.shape[0]
        y_hat = y_hat.contiguous().view(batch_size, -1)
        y = y.contiguous().view(batch_size, -1)
        if gt_y is not None:
            gt_y = gt_y.contiguous().view(batch_size, -1)
        X = X.contiguous().view(X.shape[0], -1, X.shape[-1])
        X_t = X_t.contiguous().view(X_t.shape[0], -1, X_t.shape[-1])
        feat_dim = X.shape[-1]
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x >= 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views[x]]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            if gt_y is not None and unlabeled:
                return None, None, None, None
            return None, None, None

        X_ = torch.empty((0, feat_dim)).cuda()
        X_t_ = torch.empty((0, feat_dim)).cuda()
        y_ = torch.empty(0).cuda()
        Y_ = torch.empty(0).cuda()
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            for cls_id in this_classes:
                n_view = self.max_views[cls_id]
                indices = None
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard + num_easy >= n_view and num_easy < n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_hard + num_easy >= n_view and num_hard < n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    all_indices = torch.cat([easy_indices, hard_indices])
                    num_all = num_hard + num_easy
                    perm = torch.randperm(num_all)
                    all_indices = all_indices[perm]
                    if num_all < n_view:
                        coffient = math.ceil(n_view / num_all)
                        padding_size = n_view - coffient * num_all
                        indices = all_indices.repeat(coffient, 1)
                        if not padding_size == 0:
                            indices = indices[:padding_size]
                if indices is None:
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)
                X_ = torch.cat((X_, X[ii, indices, :].squeeze(1)), dim=0)
                X_t_ = torch.cat((X_t_, X_t[ii, indices, :].squeeze(1)), dim=0)
                if gt_y is not None:
                    Y_ = torch.cat((Y_, gt_y[ii, indices].squeeze(1)))
                Y = torch.ones(n_view).cuda() * cls_id
                y_ = torch.cat((y_, Y))
        if gt_y is not None and unlabeled:
            return X_, X_t_, y_, Y_
        return X_, X_t_, y_

    def _cal_loss(self, logits, neg_logits, mask, unlabeled):
        outs = {}
        device = logits.device
        if self.mode == 'N':
            diag_mask = torch.zeros_like(mask).scatter_(1, torch.arange(logits.shape[0]).view(-1, 1).to(device), 1)
            logits_self = (diag_mask * logits).sum(dim=1)
            mask = mask - diag_mask
            logits_calmax = logits.masked_fill(~mask.bool(), -1 * 2 / self.temperature)
            logits_topn, _ = torch.topk(logits_calmax, k=self.topk, dim=1)
            logits_topn = torch.cat((logits_topn, logits_self.unsqueeze(1)), dim=1)
            # if unlabeled:
            #     topn_sum = logits_topn.sum(dim=1)
            #     sum_mean = topn_sum.mean()
            #     sum_std = torch.std(topn_sum, unbiased=False)
            #     mask_err = topn_sum < sum_mean - 3 * sum_std
            #     logits_topn = logits_topn[~mask_err]
            #     neg_logits = neg_logits[~mask_err]
            #     outs['mask'] = mask_err
            log_prob = logits_topn - torch.log(torch.exp(logits_topn) + neg_logits)
            loss = - log_prob
            loss = loss[~torch.isnan(loss)].sum() / (logits.shape[0] * (self.topk + 1))
            outs['loss1'] = 0 * loss
            outs['loss2'] = 0 * loss
        elif self.mode == 'N+A':
            diag_mask = torch.zeros_like(mask).scatter_(1, torch.arange(logits.shape[0]).view(-1, 1).to(device), 1)
            logits_self = (diag_mask * logits).sum(dim=1)
            mask = mask - diag_mask
            random_mask = torch.randint(0, 2, (logits.shape[0],), device=device).bool()
            logits_calmax = logits.masked_fill(~mask.bool(), -1 * 2 / self.temperature)
            # logits_topn = logits_calmax.max(dim=1).values.unsqueeze(1)
            logits_topn, _ = torch.topk(logits_calmax, k=self.topk, dim=1)
            logits_topn = torch.cat((logits_topn, logits_self.unsqueeze(1)), dim=1)
            log_prob = logits_topn - torch.log(torch.exp(logits_topn) + neg_logits)
            loss1 = - log_prob[random_mask]
            loss1 = loss1[~torch.isnan(loss1)].sum() / (logits.shape[0] * (self.topk + 1))
            logits = logits - torch.log(torch.exp(logits) + neg_logits)
            loss2 = - (mask * logits).sum(1) / mask.sum(1)
            loss2 = loss2[~random_mask]
            loss2 = loss2[~torch.isnan(loss2)].sum() / logits.shape[0]
            loss = loss1 + loss2
            if unlabeled:
                outs['mask'] = None
            outs['loss1'] = loss1
            outs['loss2'] = loss2
        elif self.mode == 'N+A+F':
            random_mask = torch.randint(0, 2, (logits.shape[0],), device=device)
            logits_calmax = logits.masked_fill(~mask.bool(), -1 * 2 / self.temperature)
            # logits_topn = logits_calmax.max(dim=1).values.unsqueeze(1)
            logits_topn, _ = torch.topk(logits_calmax, k=self.topk, dim=1)
            if unlabeled:
                err1 = error_mask(logits_topn).int()
                err2 = error_mask(logits).int()
                mask_err1 = (err1 * random_mask)
                mask_err2 = (err2 * (1 - random_mask))
                outs['mask'] = (mask_err1 + mask_err2).bool()
                mask_err1 = ((1 - err1) * random_mask).bool()
                mask_err2 = ((1 - err2) * (1 - random_mask)).bool()
            else:
                mask_err1 = random_mask.bool()
                mask_err2 = ~(random_mask.bool())
            log_prob = logits_topn - torch.log(torch.exp(logits_topn) + neg_logits)
            loss1 = - log_prob[~mask_err1]
            loss1 = loss1[~torch.isnan(loss1)].sum() / (logits.shape[0] * self.topk)
            logits = logits - torch.log(torch.exp(logits) + neg_logits)
            loss2 = - (mask * logits).sum(1) / mask.sum(1)
            loss2 = loss2[~mask_err2]
            loss2 = loss2[~torch.isnan(loss2)].sum() / logits.shape[0]
            loss = loss1 + loss2
            outs['loss1'] = loss1
            outs['loss2'] = loss2
        elif self.mode == 'A':
            logits = logits - torch.log(torch.exp(logits) + neg_logits)
            loss = - (mask * logits).sum(1) / mask.sum(1)
            nan_mask = torch.isnan(loss)
            loss = loss[~nan_mask]
            loss = loss.mean()
            if unlabeled:
                outs['mask'] = None
            outs['loss1'] = 0 * loss
            outs['loss2'] = 0 * loss
        outs['loss'] = loss.cuda()
        outs['loss1'] = outs['loss1'].cuda()
        outs['loss2'] = outs['loss2'].cuda()
        return outs

    def _contrastive(self, feats, feats_t, labels, unlabeled=False):
        if feats is None:
            return None
        anchor_count = feats.shape[0]
        labels = labels.contiguous().view(-1, 1)

        anchor_feature = feats
        contrast_labels = labels
        contrast_feature = feats_t

        mask = torch.eq(labels, torch.transpose(contrast_labels, 0, 1)).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        neg_mask = 1 - mask
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_count).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdims=True)
        outs = self._cal_loss(logits, neg_logits, mask, unlabeled)
        return outs

    def _contrastive_memory_bank(self, feats, feats_t, labels, unlabeled=False):
        if feats is None:
            return None
        anchor_count = feats.shape[0]
        device = feats.device
        if anchor_count > 1200:
            feats = feats.cpu()
            feats_t = feats_t.cpu()
            labels = labels.cpu()
            device = 'cpu'
            real_device = 'cuda'
        labels = labels.contiguous().view(-1, 1)

        anchor_feature = feats

        memory_bank_use = self.memory_bank is not None and len(self.memory_bank) > 0
        if memory_bank_use:
            contrast_feature_que, contrast_labels_que = self.memory_bank.sample_queue_negative()
            contrast_feature_que, contrast_labels_que = contrast_feature_que.to(device), contrast_labels_que.to(device)
            contrast_labels_que = contrast_labels_que.contiguous().view(-1, 1)

        contrast_labels = labels
        contrast_feature = feats_t
        # contrast_count = anchor_count

        mask = torch.eq(labels, torch.transpose(contrast_labels, 0, 1)).float().to(device)
        # mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        # if not memory_bank_use:
        # 跟自己的对比损失没必要计算
        # logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_count).view(-1, 1).to(device), 0)
        # 这个mask是计算i与i+的关键部分。
        # mask = mask * logits_mask
        if memory_bank_use:
            mask_que = torch.eq(labels, torch.transpose(contrast_labels_que, 0, 1)).float().to(device)
            neg_mask_que = 1 - mask_que
            mask = torch.cat([mask, mask_que], dim=1)
            neg_mask = torch.cat([neg_mask, neg_mask_que], dim=1)
            contrast_feature = torch.cat([contrast_feature, contrast_feature_que])

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdims=True)
        outs = self._cal_loss(logits, neg_logits, mask, unlabeled)
        if anchor_count > 1200:
            outs['loss'] = outs['loss'].to(real_device)
            outs['loss1'] = outs['loss1'].to(real_device)
            outs['loss1'] = outs['loss1'].to(real_device)
        return outs

    def _get_negative_mask(self, anchor_feats, labels, contrast_labels):
        dot_ = torch.mm(anchor_feats, self.memory_bank.mean_features)
        mask1 = dot_ > dot_[labels]
        contrast_labels_mask = F.one_hot(contrast_labels)
        mask = torch.mm(mask1, contrast_labels_mask)
        return mask

    def forward(self, feats, feats_t, labels, predict, unlabeled=True, gtlabels=None):
        outs = {}
        batchsize, _, h, w = feats.shape
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        if gtlabels is not None:
            gtlabels = gtlabels.unsqueeze(1).float().clone()
            gtlabels = torch.nn.functional.interpolate(gtlabels, (feats.shape[2], feats.shape[3]), mode='nearest')
            gtlabels = gtlabels.squeeze(1).long()
        predict = predict.unsqueeze(1).float().clone()
        predict = torch.nn.functional.interpolate(predict, (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        predict = predict.squeeze(1).long()
        feats = feats.permute(0, 2, 3, 1)
        feats_t = feats_t.permute(0, 2, 3, 1)
        if self.strategy == 'random':
            sampling_strategy = self._random_sampling
        elif self.strategy == 'semi-hard':
            sampling_strategy = self._semi_sampling
        else:
            sampling_strategy = self._area_sampling
        # memory_bank_use = self.memory_bank is not None and len(self.memory_bank) > 0
        if unlabeled and gtlabels is not None:
            feats_, feats_t_, labels_, gtlabels_ = sampling_strategy(feats, feats_t, labels, predict, unlabeled,
                                                                     gt_y=gtlabels)
            if labels_ is not None:
                self.eval_bank.add(labels_, gtlabels_)
        else:
            feats_, feats_t_, labels_ = self._semi_sampling(feats, feats_t, labels, predict, unlabeled)
        if feats_ is not None and feats_.shape[0] == 0:
            outs['loss'] = 0 * feats.sum()
            outs['loss1'] = 0 * feats.sum()
            outs['loss2'] = 0 * feats.sum()
            return outs
        if self.memory_bank is not None:
            outs = self._contrastive_memory_bank(feats_, feats_t_, labels_, unlabeled)
            # if unlabeled and gtlabels is not None:
            #     mask_err = outs['mask']
            #     if mask_err is not None:
            #         self.eval_bank_2.add(labels_[mask_err], gtlabels_[mask_err])
        else:
            outs = self._contrastive(feats_, feats_t_, labels_, unlabeled)
        with torch.no_grad():
            if self.memory_bank is not None:
                if unlabeled:
                    self.memory_bank.dequeue_enqueue(feats_t_, labels_, gtlabels_, unlabeled)
                else:
                    self.memory_bank.dequeue_enqueue(feats_t_, labels_, labels_, unlabeled)
        return outs


class MoCoMemoryBank:
    def __init__(self, class_num=19, memory_size=2000, pixel_update_freq=50,
                 feat_dim=128, device='cpu'):
        super(MoCoMemoryBank, self).__init__()
        self.feat_dim = feat_dim
        self.device = device
        self.seg_queue = [torch.empty((feat_dim, 0)).to(self.device) for _ in range(class_num)]
        self.gt_queue = [torch.empty((0,)).to(self.device) for _ in range(class_num)]
        self.seg_queue_ptr = torch.zeros(class_num, dtype=torch.long).to(self.device)
        self.class_num = class_num
        self.memory_size = memory_size
        self.pixel_update_freq = pixel_update_freq
        self.best_ratio = 2 * self.class_num * self.class_num
        self.mean_feature = [torch.zeros((feat_dim, self.class_num)).to(self.device) for _ in range(class_num)]
        self.queue_cos_mean = -torch.ones((self.class_num,)).to(self.device)
        self.query_hard_cos_mean = -torch.ones((self.class_num,)).to(self.device)
        self.query_easy_cos_mean = -torch.ones((self.class_num,)).to(self.device)

    def __len__(self):
        length = 0
        for i, seg_feats in enumerate(self.seg_queue):
            # if i == 0: continue
            tmp_len = seg_feats.shape[1]
            # 防止出现一个类别一直没有样本的情况
            if tmp_len == 0:
                return 0
            length += tmp_len
        return length

    def dequeue_enqueue(self, feats, labels, gtlabels=None, unlabeled=False):
        memory_size = self.memory_size
        with torch.no_grad():
            this_label_ids = torch.unique(labels)
            # this_label_ids = [x for x in this_label_ids if x > 0 and not x == 255]
            this_label_ids = [x.long() for x in this_label_ids if not x == 255]
            feats = torch.transpose(feats, 0, 1)
            for lb in this_label_ids:
                # 转化为方便处理的tensor
                idxs = (labels == lb).nonzero()

                # total area enqueue and dequeue
                # feat = torch.mean(feats[:, idxs].squeeze(2), dim=1)
                # ptr = int(self.seg_queue_ptr[lb])
                # length = self.seg_queue[lb].shape[1]
                # if length < memory_size:
                #     self.seg_queue[lb] = torch.cat((self.seg_queue[lb], F.normalize(feat, p=2, dim=0).unsqueeze(1)),
                #                                    dim=1)
                # else:
                #     self.seg_queue[lb][:, ptr] = F.normalize(feat, p=2, dim=0)
                #     self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + 1) % memory_size

                # pixel enqueue and dequeue
                # 关键的一步 之前没引用idxs
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                # 关键的一步 之前没引用idxs
                feat = feats[:, idxs[perm[:K], 0]]
                gt_ = gtlabels[idxs[perm[:K], 0]]
                ptr = int(self.seg_queue_ptr[lb])
                length = self.seg_queue[lb].shape[1]
                if length < memory_size:
                    if ptr + K >= memory_size:
                        self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                        self.seg_queue[lb] = self.seg_queue[lb][:, -memory_size:]
                        self.seg_queue_ptr[lb] = 0
                        if gtlabels is not None:
                            self.gt_queue[lb] = torch.cat((self.gt_queue[lb], gt_))
                            self.gt_queue[lb] = self.gt_queue[lb][-memory_size:]
                    else:
                        self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                        self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size
                        if gtlabels is not None:
                            self.gt_queue[lb] = torch.cat((self.gt_queue[lb], gt_))
                else:
                    if ptr + K >= memory_size:
                        self.seg_queue[lb][:, -K:] = feat
                        self.seg_queue_ptr[lb] = 0
                        if gtlabels is not None:
                            self.gt_queue[lb][-K:] = gt_
                    else:
                        self.seg_queue[lb][:, ptr:ptr + K] = feat
                        self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size
                        if gtlabels is not None:
                            self.gt_queue[lb][ptr:ptr + K] = gt_

        if self.__len__() > 0:
            self.mean_feature = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in self.seg_queue], dim=1)
            ratio = (torch.matmul(torch.transpose(self.mean_feature, 0, 1),
                                  self.mean_feature)).sum() + self.class_num * (self.class_num - 1)
            is_queue = True
            with open('3.txt', 'a') as file:
                file.write('separation active {}:{}\n'.format(is_queue, ratio))
            return ratio, True
        else:
            return self.best_ratio, False

    def sample_queue_negative(self):
        X_ = torch.cat([l for l in self.seg_queue if l.shape[1] > 0], dim=1)
        X_ = torch.transpose(X_, 0, 1)
        y_ = torch.cat([i * torch.ones(self.seg_queue[i].shape[1]) for i in range(self.class_num) if
                        self.seg_queue[i].shape[1] > 0])
        return X_, y_


class EvalModule:
    def __init__(self, nclass, feat_dim, device='cpu'):
        self.device = device
        self.nclass = nclass
        self.feat_dim = feat_dim
        self.store_features = [torch.empty(self.feat_dim, 0).to(self.device) for _ in range(self.nclass)]
        self.num_features = [0 for _ in range(self.nclass)]
        self.last_mean_features = torch.zeros((self.nclass, self.feat_dim)).to(self.device)
        self.last_mean_features_stored = False
        # self.var_features = [0 for _ in range(nclass)]

    def add(self, features, labels):
        batchsize = features.shape[0]
        for i in range(self.nclass):
            for j in range(batchsize):
                num_i = (labels[j] == i).int().sum()
                if num_i > 0:
                    self.store_features[i] = torch.cat([self.store_features[i],
                                                        torch.mean(features[j][:, (labels[j] == i).squeeze()], dim=1,
                                                                   keepdim=True)], dim=1)
                self.num_features[i] += num_i

    def indicator(self):
        mean_features = torch.stack([F.normalize(torch.mean(feats, dim=1), dim=0) for feats in self.store_features])
        separate_ratio = (torch.matmul(mean_features, torch.transpose(mean_features, 0, 1))).sum()
        var_features = [0 for _ in range(self.nclass)]
        for i, features_i in enumerate(self.store_features):
            var_features[i] = torch.pow((features_i - mean_features[i, :].unsqueeze(dim=1)), 2).sum() / self.feat_dim
        cons = [1 for _ in range(self.nclass)]
        if self.last_mean_features_stored:
            for i in range(self.nclass):
                cons[i] = (self.last_mean_features[i] * mean_features[i]).sum()
        else:
            self.last_mean_features_stored = True
        self.last_mean_features = mean_features
        self.store_features = [torch.empty(self.feat_dim, 0).to(self.device) for _ in range(self.nclass)]
        self.num_features = [0 for _ in range(self.nclass)]
        return separate_ratio, var_features, cons


class EvalFeat:
    def __init__(self):
        self.right = 0
        self.all = 0

    def add(self, pred, gt):
        delete_ignore_ids = gt != 255
        gt = gt[delete_ignore_ids]
        pred = pred[delete_ignore_ids]
        length = gt.shape[0]
        tmp_right = (pred == gt).sum()
        self.all += length
        self.right += tmp_right

    def indicator(self):
        try:
            accuracy = self.right / self.all
        except:
            accuracy = 0
        self.all = 0
        self.right = 0
        return accuracy


def edge_interior(label_mask: torch.Tensor, ignore_mask: torch.Tensor, min_k=3, max_k=5, pow=3, edge=5):
    device = label_mask.device
    label_mask = label_mask.cpu().numpy()
    ignore_mask = ignore_mask.cpu().numpy()
    label_unique = np.unique(label_mask)
    label_areas = {label: (np.sum(label_mask == label)) ** (1 / pow) for label in label_unique}
    min_area = min(label_areas.values())
    max_area = max(label_areas.values())
    mask_edges = []
    mask_inters = []
    for idx, label in enumerate(label_unique):
        if label == 255:
            continue
        if len(label_unique) == 1:
            k = min_k
        else:
            # norm_area = (max_area - label_areas[label]) / (max_area - min_area)
            k = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_eq = (label_mask == label).astype('uint8')
        # mask_dilate = cv2.dilate(mask_eq, kernel) - mask_eq
        mask_inter = cv2.erode(mask_eq, kernel)
        mask_edge = mask_eq - mask_inter
        mask_inter[:edge, :], mask_inter[-edge:, :], mask_inter[:, :edge], mask_inter[:, -edge:] = 0, 0, 0, 0
        mask_edge[:edge, :], mask_edge[-edge:, :], mask_edge[:, :edge], mask_edge[:, -edge:] = 0, 0, 0, 0
        mask_inter[ignore_mask] = 0
        mask_edge[ignore_mask] = 0
        mask_edge = torch.from_numpy(mask_edge.astype('int32')).to(device).bool()
        mask_inter = torch.from_numpy(mask_inter.astype('int32')).to(device).bool()
        mask_edges.append(mask_edge)
        mask_inters.append(mask_inter)

    return mask_edges, mask_inters


def error_mask(tensor: torch.Tensor, N=2):
    tensor_sum = tensor.sum(dim=1)
    sum_mean = tensor_sum.mean()
    sum_std = torch.std(tensor_sum, unbiased=False)
    mask_error = tensor_sum < sum_mean - N * sum_std
    return mask_error


def plot_color(tensor):
    gray_array = tensor.cpu().numpy()
    color_mapping = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]
    ]
    color_map = np.array(color_mapping)
    gray_array[gray_array == 255] = 0
    color_image = color_map[gray_array]
    color_image = Image.fromarray(color_image.astype('uint8'))
    color_image.save('./result.png')
