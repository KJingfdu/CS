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
    def __init__(self, nclass, temperature=0.1,
                 neg_num=64, memory_bank=True, pixel_update_freq=50,
                 memory_size=2000, small_area=True,
                 feat_dim=256, max_positive=False,
                 device='cpu', ignore_label=255,
                 eval_feat=True):
        super(MocoContrastLoss, self).__init__()
        self.temperature = temperature
        self.max_positive = max_positive
        self.base_temperature = temperature
        self.ignore_label = ignore_label
        self.nclass = nclass
        self.max_samples = int(neg_num * nclass)
        self.max_views = [neg_num for _ in range(nclass)]
        self.small_area = small_area
        if memory_bank:
            self.memory_bank = MoCoMemoryBank(nclass, memory_size, pixel_update_freq, feat_dim, device)
            self.use_sds = False
        else:
            self.memory_bank = None
        if eval_feat:
            self.eval_bank = EvalFeat()

    def _random_sampling(self, X, X_t, y_hat, y, unlabeled=True, gt_y=None):
        batch_size = X.shape[0]
        y_hat = y_hat.contiguous().view(batch_size, -1)
        # y = y.contiguous().view(batch_size, -1)
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
            return None, None

        X_ = torch.empty((0, feat_dim)).cuda()
        X_t_ = torch.empty((0, feat_dim)).cuda()
        y_ = torch.empty(0).cuda()
        Y_ = torch.empty(0).cuda()
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            # this_y = y[ii]
            this_classes = classes[ii]
            for cls_id in this_classes:
                n_view = self.max_views[cls_id]
                indices = (this_y_hat == cls_id).nonzero()
                num_all = indices.shape[0]
                perm = torch.randperm(num_all)
                indices = indices[perm[:n_view]]

                X_ = torch.cat((X_, X[ii, indices, :].squeeze(1)), dim=0)
                X_t_ = torch.cat((X_t_, X_t[ii, indices, :].squeeze(1)), dim=0)
                if gt_y is not None:
                    Y_ = torch.cat((Y_, gt_y[ii, indices].squeeze(1)))
                Y = torch.ones(n_view).cuda() * cls_id
                y_ = torch.cat((y_, Y))
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
            this_y = y[ii]
            # 根据student model预测出的y进行mask的挑选,进而挑选hard-easy sample.
            this_y_s = y_hat[ii]
            # 图片中存在255的部分, 因为采样是根据预测结果进行的, 很可能会涉及255的部分
            ignore_mask = (this_y_s == 255)
            this_X = X[ii]
            this_X_t = X_t[ii]
            this_y_ids = this_y.unique()
            this_y_ids = [i for i in this_y_ids if not i == 255]
            edge_mask, interior_mask = edge_interior(this_y)
            # ignore的部分不予采样
            edge_mask[ignore_mask] == 0
            interior_mask[ignore_mask] == 0
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
                y_ = torch.cat((y_, this_y_s[edge_mask[id]][random_id, :]))
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
                Y = torch.ones(num_keep).cuda() * value
                y_ = torch.cat((y_, Y))
                if gt_y is not None and unlabeled:
                    this_y = gt_y[ii]
                    Y_ = torch.cat((Y_, this_y[interior_mask[id]][random_id]))
        if gt_y is not None and unlabeled:
            return X_, X_t_, y_, Y_
        return X_, X_t_, y_

    def _mid_sampling(self, X, X_t, y_hat, y, unlabeled=True, gt_y=None, decay=0.5):
        batch_size = X.shape[0]
        y_hat = y_hat.contiguous().view(batch_size, -1)
        y = y.contiguous().view(batch_size, -1)
        if gt_y is not None:
            gt_y = gt_y.contiguous().view(batch_size, -1)
        X = X.contiguous().view(X.shape[0], -1, X.shape[-1])
        X_t = X_t.contiguous().view(X_t.shape[0], -1, X_t.shape[-1])
        # y_hat为真实标签 y为预测标签
        feat_dim = X.shape[-1]
        classes = []
        total_classes = 0
        if self.use_sds and unlabeled:
            mean_features = self.memory_bank.mean_feature
        # filter each image, to find what class they have num > self.max_view pixel
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            # this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if x >= 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views[x]]
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        # n_view = self.max_samples // total_classes
        # n_view = min(n_view, self.max_views)
        # X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        X_ = torch.empty((0, feat_dim)).cuda()
        X_t_ = torch.empty((0, feat_dim)).cuda()
        y_ = torch.empty(0).cuda()
        if gt_y is not None:
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
                if self.use_sds and unlabeled:
                    mean_feature_i = mean_features[:, cls_id]  # (feat_dim,)
                    hard_features = X_t[ii, hard_indices, :].squeeze(1)
                    easy_features = X_t[ii, easy_indices, :].squeeze(1)
                    # feature_cos_seg = torch.mm(self.memory_bank.seg_queue[cls_id].transpose(1, 0),
                    #                            mean_feature_i.unsqueeze(1)).squeeze()
                    feature_cos_hard = torch.mm(hard_features, mean_feature_i.unsqueeze(1)).squeeze()
                    feature_cos_easy = torch.mm(easy_features, mean_feature_i.unsqueeze(1)).squeeze()
                    # mean_seg = feature_cos_seg.mean()
                    mean_hard = feature_cos_hard.mean()
                    # mean_easy = feature_cos_easy.mean()
                    if not torch.isnan(mean_hard):
                        self.memory_bank.query_hard_cos_mean[cls_id] = decay * self.memory_bank.query_hard_cos_mean[
                            cls_id] + (1 - decay) * mean_hard
                    # if not torch.isnan(mean_easy):
                    #     self.memory_bank.query_easy_cos_mean[cls_id] = decay * self.memory_bank.query_easy_cos_mean[
                    #         cls_id] + (1 - decay) * mean_easy
                    separ_hard_indices = feature_cos_hard >= self.memory_bank.query_hard_cos_mean[cls_id]
                    separ_easy_indices = feature_cos_easy >= self.memory_bank.query_hard_cos_mean[cls_id]
                    if not len(separ_hard_indices.shape):
                        separ_hard_indices = separ_hard_indices.unsqueeze(0)
                    if not len(separ_easy_indices.shape):
                        separ_easy_indices = separ_easy_indices.unsqueeze(0)
                    hard_indices = hard_indices[separ_hard_indices]
                    easy_indices = easy_indices[separ_easy_indices]
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
                    if num_all < n_view and num_all > 0:
                        coffient = math.ceil(n_view / num_all)
                        padding_size = n_view - coffient * num_all
                        indices = all_indices.repeat(coffient, 1)
                        if not padding_size == 0:
                            indices = indices[:padding_size]
                    else:
                        continue
                    # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    # raise Exception
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

    def _semi_sampling(self, X, X_t, y_hat, y, unlabeled=True, gt_y=None):
        batch_size = X.shape[0]
        y_hat = y_hat.contiguous().view(batch_size, -1)
        y = y.contiguous().view(batch_size, -1)
        if gt_y is not None:
            gt_y = gt_y.contiguous().view(batch_size, -1)
        X = X.contiguous().view(X.shape[0], -1, X.shape[-1])
        X_t = X_t.contiguous().view(X_t.shape[0], -1, X_t.shape[-1])
        # y_hat为真实标签 y为预测标签
        feat_dim = X.shape[-1]
        classes = []
        total_classes = 0
        # filter each image, to find what class they have num > self.max_view pixel
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            # this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if x >= 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views[x]]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        # n_view = self.max_samples // total_classes
        # n_view = min(n_view, self.max_views)

        # X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
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
                    # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    # raise Exception
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

    def _contrastive(self, feats, labels):
        if feats is None:
            return None
        anchor_count = feats.shape[0]
        labels = labels.contiguous().view(-1, 1)

        anchor_feature = feats
        contrast_labels = labels
        contrast_feature = anchor_feature

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
        if self.max_positive:
            logits = logits.masked_fill(~mask.bool(), -1 * 2 / self.temperature)
            logits = logits.max(dim=1).values.unsqueeze(1)
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits + neg_logits)
            loss = - (self.temperature / self.base_temperature) * log_prob
        else:
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits + neg_logits)
            # sum(1)是为了按照query的i来进行计算损失 现在计算出来的一列向量，每一列都是一个query的对比损失 最后计算平均得出损失函数
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        nan_mask = torch.isnan(loss)
        loss = loss[~nan_mask]
        loss = loss.mean()

        return loss

    def _contrastive_memory_bank(self, feats, feats_t, labels):
        # 解释代码中的anchor_count和contrast_count
        # 原始代码来自于ContrastiveSeg库。 2021ICCV
        # 由于源代码在选取hard特征时，是选取了每个类别固定长度n_view个点的特征，其标签都被定义为一个数值。所以后续需要repeat。
        # 然而在保存各个类别的特征及其sampling时，得出的特征和类别信息一一对应，不需要repeat，所以count为1
        if feats is None:
            return None
        anchor_count = feats.shape[0]
        device = feats.device
        # if anchor_count > 1200:
        #     feats = feats.cpu()
        #     feats_t = feats_t.cpu()
        #     labels = labels.cpu()
        #     device = 'cpu'
        #     real_device = 'cuda'
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
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_count).view(-1, 1).to(device), 0)
        # 这个mask是计算i与i+的关键部分。
        mask = mask * logits_mask
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
        if self.max_positive:
            logits_calmax = logits.masked_fill(~mask.bool(), -1 * 2 / self.temperature)
            # logits_topn = logits_calmax.max(dim=1).values.unsqueeze(1)
            logits_topn, _ = torch.topk(logits_calmax, k=3, dim=1)
            log_prob = logits_topn - torch.log(torch.exp(logits_topn) + neg_logits)
            loss = - (self.temperature / self.base_temperature) * log_prob
        else:
            logits = logits - torch.log(torch.exp(logits) + neg_logits)
            loss = - (self.temperature / self.base_temperature) * (mask * logits).sum(1) / mask.sum(1)
        nan_mask = torch.isnan(loss)
        loss = loss[~nan_mask]
        loss = loss.mean()
        # if anchor_count > 1200:
        #     loss = loss.to(real_device)
        return loss

    def forward(self, feats, feats_t, labels, predict, unlabeled=True, gtlabels=None):
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
        # memory_bank_use = self.memory_bank is not None and len(self.memory_bank) > 0
        if unlabeled and gtlabels is not None:
            feats_, feats_t_, labels_, gtlabels_ = self._area_sampling(feats, feats_t, labels, predict, unlabeled,
                                                                       gt_y=gtlabels)
            self.eval_bank.add(labels_, gtlabels_)
        else:
            feats_, feats_t_, labels_ = self._area_sampling(feats, feats_t, labels, predict, unlabeled)
        if feats_.shape[0] == 0:
            loss = 0 * feats.sum()
            return loss
        if self.memory_bank is not None:
            loss = self._contrastive_memory_bank(feats_, feats_t_, labels_)
        else:
            loss = self._contrastive(feats_, labels_)
        with torch.no_grad():
            if self.memory_bank is not None:
                sperate_ratio, self.use_sds = self.memory_bank.mid_dequeue_enqueue(feats_t, labels, unlabeled)
                if self.memory_bank.best_ratio > sperate_ratio:
                    self.memory_bank.best_ratio = sperate_ratio
        return loss


class MoCoMemoryBank:
    def __init__(self, class_num=19, memory_size=2000, pixel_update_freq=50,
                 feat_dim=128, device='cpu', split=1 / 2):
        super(MoCoMemoryBank, self).__init__()
        self.feat_dim = feat_dim
        self.device = device
        self.seg_queue = [torch.empty((feat_dim, 0)).to(self.device) for _ in range(class_num)]
        self.seg_queue_ptr = torch.zeros(class_num, dtype=torch.long).to(self.device)
        self.class_num = class_num
        self.memory_size = memory_size
        self.pixel_update_freq = pixel_update_freq
        self.best_ratio = 2 * self.class_num * self.class_num
        self.split = split
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

    def random_dequeue_enqueue(self, feats, labels, SMALL_AREA=True):
        batch_size, H, W, feat_dim = feats.shape
        memory_size = self.memory_size
        with torch.no_grad():
            for bs in range(batch_size):
                this_feat = feats[bs].contiguous().to(self.device)
                this_label = labels[bs].contiguous().to(self.device)
                this_label_ids = torch.unique(this_label)
                # this_label_ids = [x for x in this_label_ids if x > 0 and not x == 255]
                this_label_ids = [x for x in this_label_ids if not x == 255]

                for lb in this_label_ids:
                    if SMALL_AREA:
                        # 这些步骤先来得出mask的具体位置
                        idxs = (this_label == lb).nonzero()
                        mask = torch.zeros_like(this_label).to(self.device)
                        mask[idxs[:, 0], idxs[:, 1]] = 1
                        mask_list = small_area(mask, self.pixel_update_freq)

                    # 转化为方便处理的tensor
                    this_label_s = this_label.view(-1)
                    idxs = (this_label_s == lb).nonzero()
                    this_feat_s = torch.transpose(this_feat.view(-1, feat_dim), 0, 1)

                    if SMALL_AREA:
                        # small area enqueue and dequeue
                        for mask in mask_list:
                            feat = torch.mean(this_feat_s[:, mask], dim=1).squeeze(1)
                            ptr = int(self.seg_queue_ptr[lb])
                            length = self.seg_queue[lb].shape[1]
                            if length < memory_size:
                                self.seg_queue[lb] = torch.cat(
                                    (self.seg_queue[lb], F.normalize(feat, p=2, dim=0).unsqueeze(1)),
                                    dim=1)
                            else:
                                self.seg_queue[lb][:, ptr] = F.normalize(feat, p=2, dim=0)
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + 1) % memory_size
                        # to balance local info and context info

                    # pixel enqueue and dequeue
                    num_pixel = idxs.shape[0]
                    perm = torch.randperm(num_pixel)
                    K = min(num_pixel, self.pixel_update_freq)
                    # 关键的一步 之前没引用idxs
                    feat = this_feat_s[:, idxs[perm[:K], 0]]
                    ptr = int(self.seg_queue_ptr[lb])
                    length = self.seg_queue[lb].shape[1]
                    if length < memory_size:
                        if ptr + K >= memory_size:
                            self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                            self.seg_queue[lb] = self.seg_queue[lb][:, -memory_size:]
                            self.seg_queue_ptr[lb] = 0
                        else:
                            self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size
                    else:
                        if ptr + K >= memory_size:
                            self.seg_queue[lb][:, -K:] = feat
                            self.seg_queue_ptr[lb] = 0
                        else:
                            self.seg_queue[lb][:, ptr:ptr + K] = feat
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size
        if self.__len__() > 0:
            self.mean_feature = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in self.seg_queue], dim=1)
            ratio = (torch.matmul(torch.transpose(self.mean_feature, 0, 1),
                                  self.mean_feature)).sum() + self.class_num * (self.class_num - 1)
            is_queue = True
            with open('3.txt', 'a') as file:
                file.write('separation random {}:{}\n'.format(is_queue, ratio))
            return ratio, True
        else:
            return self.best_ratio, False

    def dequeue_enqueue(self, feats, labels, SMALL_AREA=True):
        memory_queue = [a.clone() for a in self.seg_queue]
        memory_queue_ptr = self.seg_queue_ptr.clone()
        batch_size, H, W, feat_dim = feats.shape
        memory_size = self.memory_size
        with torch.no_grad():
            for bs in range(batch_size):
                this_feat = feats[bs].contiguous().to(self.device)
                this_label = labels[bs].contiguous().to(self.device)
                this_label_ids = torch.unique(this_label)
                # this_label_ids = [x for x in this_label_ids if x > 0 and not x == 255]
                this_label_ids = [x for x in this_label_ids if not x == 255]

                for lb in this_label_ids:
                    if SMALL_AREA:
                        # 这些步骤先来得出mask的具体位置
                        idxs = (this_label == lb).nonzero()
                        mask = torch.zeros_like(this_label).to(self.device)
                        mask[idxs[:, 0], idxs[:, 1]] = 1
                        mask_list = small_area(mask, self.pixel_update_freq)

                    # 转化为方便处理的tensor
                    this_label_s = this_label.view(-1)
                    idxs = (this_label_s == lb).nonzero()
                    this_feat_s = torch.transpose(this_feat.view(-1, feat_dim), 0, 1)

                    # total area enqueue and dequeue
                    feat = torch.mean(this_feat_s[:, idxs], dim=1).squeeze(1)
                    ptr = int(self.seg_queue_ptr[lb])
                    length = self.seg_queue[lb].shape[1]
                    if length < memory_size:
                        self.seg_queue[lb] = torch.cat((self.seg_queue[lb], F.normalize(feat, p=2, dim=0).unsqueeze(1)),
                                                       dim=1)
                    else:
                        self.seg_queue[lb][:, ptr] = F.normalize(feat, p=2, dim=0)
                    self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + 1) % memory_size

                    if SMALL_AREA:
                        # small area enqueue and dequeue
                        for mask in mask_list:
                            feat = torch.mean(this_feat_s[:, mask], dim=1).squeeze(1)
                            ptr = int(self.seg_queue_ptr[lb])
                            length = self.seg_queue[lb].shape[1]
                            if length < memory_size:
                                self.seg_queue[lb] = torch.cat(
                                    (self.seg_queue[lb], F.normalize(feat, p=2, dim=0).unsqueeze(1)),
                                    dim=1)
                            else:
                                self.seg_queue[lb][:, ptr] = F.normalize(feat, p=2, dim=0)
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + 1) % memory_size
                        # to balance local info and context info

                    # pixel enqueue and dequeue
                    num_pixel = idxs.shape[0]
                    perm = torch.randperm(num_pixel)
                    K = min(num_pixel, self.pixel_update_freq)
                    # 关键的一步 之前没引用idxs
                    feat = this_feat_s[:, idxs[perm[:K], 0]]
                    ptr = int(self.seg_queue_ptr[lb])
                    length = self.seg_queue[lb].shape[1]
                    if length < memory_size:
                        if ptr + K >= memory_size:
                            self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                            self.seg_queue[lb] = self.seg_queue[lb][:, -memory_size:]
                            self.seg_queue_ptr[lb] = 0
                        else:
                            self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size
                    else:
                        if ptr + K >= memory_size:
                            self.seg_queue[lb][:, -K:] = feat
                            self.seg_queue_ptr[lb] = 0
                        else:
                            self.seg_queue[lb][:, ptr:ptr + K] = feat
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size

        if self.__len__() > 0:
            separate, ratio, self.mean_feature = separation(self.seg_queue, memory_queue,
                                                            self.class_num * (self.class_num - 1),
                                                            self.best_ratio,
                                                            n=self.class_num * (self.class_num - 1))
            length_now = 0
            for i in range(len(self.seg_queue)):
                length_now += self.seg_queue[i].shape[1]
            length_old = 0
            for i in range(len(memory_queue)):
                length_old += memory_queue[i].shape[1]
            if separate and length_now >= length_old:
                is_queue = True
            else:
                self.seg_queue = memory_queue
                self.seg_queue_ptr = memory_queue_ptr.clone()
                is_queue = False
            with open('3.txt', 'a') as file:
                file.write('separation {}:{}\n'.format(is_queue, ratio))
            return ratio, True
        else:
            return self.best_ratio, False

    def active_dequeue_enqueue(self, feats, labels, SMALL_AREA=True, lr=5e-2, iters=1):
        batch_size, H, W, feat_dim = feats.shape
        memory_size = self.memory_size
        is_active = False
        if self.__len__() > 0:
            label_ = torch.unique(labels).tolist()
            feats_mean = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in self.seg_queue], dim=1)
            feats_mean = F.normalize(feats_mean, dim=0)
            optimized_feats = gradient_descent(feats_mean, label_, lr, iters)
            is_active = True
        with torch.no_grad():
            for bs in range(batch_size):
                this_feat = feats[bs].contiguous().to(self.device)
                this_label = labels[bs].contiguous().to(self.device)
                this_label_ids = torch.unique(this_label)
                # this_label_ids = [x for x in this_label_ids if x > 0 and not x == 255]
                this_label_ids = [x for x in this_label_ids if not x == 255]

                for lb in this_label_ids:
                    if SMALL_AREA:
                        # 这些步骤先来得出mask的具体位置
                        idxs = (this_label == lb).nonzero()
                        mask = torch.zeros_like(this_label).to(self.device)
                        mask[idxs[:, 0], idxs[:, 1]] = 1
                        mask_list = small_area(mask, self.pixel_update_freq)

                    # 转化为方便处理的tensor
                    this_label_s = this_label.view(-1)
                    idxs = (this_label_s == lb).nonzero()
                    this_feat_s = torch.transpose(this_feat.view(-1, feat_dim), 0, 1)

                    # total area enqueue and dequeue
                    feat_total = torch.mean(this_feat_s[:, idxs], dim=1)

                    if SMALL_AREA:
                        # small area enqueue and dequeue
                        for mask in mask_list:
                            feat = torch.mean(this_feat_s[:, mask], dim=1)
                            feat_total = torch.cat([feat_total, feat], dim=1)

                        # to balance local info and context info

                    # 关键的一步 之前没引用idxs
                    feat_total = torch.cat([feat_total, this_feat_s[:, idxs[:, 0]]], dim=1)
                    ptr = int(self.seg_queue_ptr[lb])
                    length = self.seg_queue[lb].shape[1]

                    num_pixel = feat_total.shape[1]
                    device = feat_total.device
                    K = min(num_pixel, self.pixel_update_freq)
                    # 在这里计算与最理想的分离方向最相近的向量
                    if is_active:
                        feat_cos = torch.mm(optimized_feats[:, lb].unsqueeze(0), feat_total).squeeze()
                        # feat_total = feat_total[:, feat_cos >= feat_cos.mean()]
                        # num_cos = feat_total.shape[1]
                        # K = min(K, num_cos)
                        # perm = torch.randperm(num_cos)
                        # feat = feat_total[:, perm[:K]]
                        _, index = torch.topk(feat_cos, k=K, dim=0)
                        feat = feat_total[:, index]
                    else:
                        perm = torch.randperm(num_pixel)
                        # 关键的一步 之前没引用idxs
                        feat = feat_total[:, perm[:K]]
                    if length < memory_size:
                        if ptr + K >= memory_size:
                            self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                            self.seg_queue[lb] = self.seg_queue[lb][:, -memory_size:]
                            self.seg_queue_ptr[lb] = 0
                        else:
                            self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size
                    else:
                        if ptr + K >= memory_size:
                            self.seg_queue[lb][:, -K:] = feat
                            self.seg_queue_ptr[lb] = 0
                        else:
                            self.seg_queue[lb][:, ptr:ptr + K] = feat
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size

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

    def mid_dequeue_enqueue(self, feats, labels, unlabeled=True, decay=0.5):
        batch_size, H, W, feat_dim = feats.shape
        memory_size = self.memory_size
        is_active = False
        if self.__len__() > 0 and unlabeled:
            feats_mean = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in self.seg_queue], dim=1)
            feats_mean = F.normalize(feats_mean, dim=0)
            is_active = True
        with torch.no_grad():
            for bs in range(batch_size):
                this_feat = feats[bs].contiguous().to(self.device)
                this_label = labels[bs].contiguous().to(self.device)
                this_label_ids = torch.unique(this_label)
                # this_label_ids = [x for x in this_label_ids if x > 0 and not x == 255]
                this_label_ids = [x for x in this_label_ids if not x == 255]

                for lb in this_label_ids:
                    # 转化为方便处理的tensor
                    this_label_s = this_label.view(-1)
                    idxs = (this_label_s == lb).nonzero()
                    this_feat_s = torch.transpose(this_feat.view(-1, feat_dim), 0, 1)

                    # total area enqueue and dequeue
                    feat_total = torch.mean(this_feat_s[:, idxs], dim=1)

                    # 关键的一步 之前没引用idxs
                    feat_total = torch.cat([feat_total, this_feat_s[:, idxs[:, 0]]], dim=1)
                    ptr = int(self.seg_queue_ptr[lb])
                    length = self.seg_queue[lb].shape[1]

                    num_pixel = feat_total.shape[1]
                    K = min(num_pixel, self.pixel_update_freq)
                    # 在这里计算与最理想的分离方向最相近的向量
                    if is_active and unlabeled:
                        # feat_cos_lb = torch.mm(feats_mean[:, lb].unsqueeze(0), self.seg_queue[lb]).squeeze()
                        feat_cos = torch.mm(feats_mean[:, lb].unsqueeze(0), feat_total).squeeze()
                        self.queue_cos_mean[lb] = decay * self.queue_cos_mean[lb] + (1 - decay) * feat_cos.mean()
                        feat_total = feat_total[:, feat_cos >= self.queue_cos_mean[lb]]
                        num_cos = feat_total.shape[1]
                        K = min(K, num_cos)
                        perm = torch.randperm(num_cos)
                        feat = feat_total[:, perm[:K]]
                        # _, index = torch.topk(feat_cos, k=K, dim=0)
                        # feat = feat_total[:, index]
                    else:
                        perm = torch.randperm(num_pixel)
                        # 关键的一步 之前没引用idxs
                        feat = feat_total[:, perm[:K]]
                    if length < memory_size:
                        if ptr + K >= memory_size:
                            self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                            self.seg_queue[lb] = self.seg_queue[lb][:, -memory_size:]
                            self.seg_queue_ptr[lb] = 0
                        else:
                            self.seg_queue[lb] = torch.cat((self.seg_queue[lb], feat), dim=1)
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size
                    else:
                        if ptr + K >= memory_size:
                            self.seg_queue[lb][:, -K:] = feat
                            self.seg_queue_ptr[lb] = 0
                        else:
                            self.seg_queue[lb][:, ptr:ptr + K] = feat
                            self.seg_queue_ptr[lb] = (self.seg_queue_ptr[lb] + K) % memory_size

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

    def _mean_feature(self):
        X_ = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in self.seg_queue], dim=1)
        return X_


def small_area(mask, N=50):
    h, w = mask.shape
    area = mask.sum()
    if area == h * w:
        mask_list = [mask.view(-1, 1)]
        return mask_list
    device = mask.device
    mask_list = []
    mask_dilate = mask
    while area > 1 and len(mask_list) < N:
        # 这里的3*3的腐蚀核与上面的area>1有关的，不然会导致出错
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_dilate_np = cv2.erode(mask_dilate.int().cpu().numpy().astype('uint8'), kernel).astype('int32')
        mask_dilate = (torch.from_numpy(mask_dilate_np)).to(device).float()
        tmp = (mask_dilate.view(-1) == 1).nonzero()
        mask_list.append(tmp) if len(tmp) > 0 else None
        tmp = ((mask - mask_dilate).view(-1) == 1).nonzero()
        mask_list.append(tmp) if len(tmp) > 0 else None
        area = mask_dilate.sum()
    return mask_list


class UnlabelFilter:
    def __init__(self, total_iterations, start_value=0.1, end_value=0.9, method='log'):
        self.total_iters = total_iterations
        self.current_iter = 0
        self.method = method
        self.start_value = start_value
        self.end_value = end_value

    def get_parameter(self):
        if self.method == 'exp':
            decay_rate = self.end_value / self.start_value
            parameter = self.start_value * (decay_rate ** (self.current_iter / self.total_iters))
        elif self.method == 'log':
            parameter = self.start_value + (
                    (self.end_value - self.start_value) * math.log(1 + self.current_iter) / math.log(
                1 + self.total_iters))
        else:
            progress = self.current_iter / self.total_iters
            parameter = (1 - progress) * self.start_value + progress * self.end_value
        self.current_iter += 1
        return parameter

    def __bool__(self):
        return True


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
        accuracy = self.right / self.all
        self.all = 0
        self.right = 0
        return accuracy


def edge_interior(label_mask: torch.Tensor, min_k=3, max_k=5, pow=3, edge=5):
    device = label_mask.device
    label_mask = label_mask.cpu().numpy()
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
            norm_area = (max_area - label_areas[label]) / (max_area - min_area)
            k = int(min_k + norm_area * (max_k - min_k))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_eq = (label_mask == label).astype('uint8')
        # mask_dilate = cv2.dilate(mask_eq, kernel) - mask_eq
        mask_inter = cv2.erode(mask_eq, kernel)
        mask_edge = mask_eq - mask_inter
        mask_inter[:edge, :], mask_inter[-edge:, :], mask_inter[:, :edge], mask_inter[:, -edge:] = 0, 0, 0, 0
        mask_edge[:edge, :], mask_edge[-edge:, :], mask_edge[:, :edge], mask_edge[:, -edge:] = 0, 0, 0, 0
        mask_edge = torch.from_numpy(mask_edge.astype('int32')).to(device).bool()
        mask_inter = torch.from_numpy(mask_inter.astype('int32')).to(device).bool()
        mask_edges.append(mask_edge)
        mask_inters.append(mask_inter)

    return mask_edges, mask_inters


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
