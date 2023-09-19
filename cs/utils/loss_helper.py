import numpy as np
import scipy.ndimage as nd
import math
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import dequeue_and_enqueue


def compute_rce_loss(predict, target):
    from einops import rearrange

    predict = F.softmax(predict, dim=1)

    with torch.no_grad():
        _, num_cls, h, w = predict.shape
        temp_tar = target.clone()
        temp_tar[target == 255] = 0

        label = (
            F.one_hot(temp_tar.clone().detach(), num_cls).float().cuda()
        )  # (batch, h, w, num_cls)
        label = rearrange(label, "b h w c -> b c h w")
        label = torch.clamp(label, min=1e-4, max=1.0)

    rce = -torch.sum(predict * torch.log(label), dim=1) * (target != 255).bool()
    return rce.sum() / (target != 255).sum()


def compute_unsupervised_loss(predict, target, percent, pred_teacher):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        weight = batch_size * h * w / torch.sum(target != 255)

    loss = weight * F.cross_entropy(predict, target, ignore_index=255)  # [10, 321, 321]

    return loss


def compute_contra_memobank_loss(
        rep,
        label_l,
        label_u,
        prob_l,
        prob_u,
        low_mask,
        high_mask,
        cfg,
        memobank,
        queue_prtlis,
        queue_size,
        rep_teacher,
        momentum_prototype=None,
        i_iter=0,
):
    # current_class_threshold: delta_p (0.3)
    # current_class_negative_threshold: delta_n (1)
    current_class_threshold = cfg["current_class_threshold"]
    current_class_negative_threshold = cfg["current_class_negative_threshold"]
    low_rank, high_rank = cfg["low_rank"], cfg["high_rank"]
    temp = cfg["temperature"]
    num_queries = cfg["num_queries"]
    num_negatives = cfg["num_negatives"]

    num_feat = rep.shape[1]
    num_labeled = label_l.shape[0]
    num_segments = label_l.shape[1]

    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    seg_feat_all_list = []
    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class

    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(
        0, 2, 3, 1
    )  # (num_unlabeled, h, w, num_cls)

    prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

    valid_classes = []
    new_keys = []
    for i in range(num_segments):
        low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
        high_valid_pixel_seg = high_valid_pixel[:, i]

        prob_seg = prob[:, i, :, :]
        rep_mask_low_entropy = (
                                       prob_seg > current_class_threshold
                               ) * low_valid_pixel_seg.bool()
        rep_mask_high_entropy = (
                                        prob_seg < current_class_negative_threshold
                                ) * high_valid_pixel_seg.bool()

        seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])
        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

        # positive sample: center of the class
        seg_proto_list.append(
            torch.mean(
                rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
            )
        )

        # generate class mask for unlabeled data
        # prob_i_classes = prob_indices_u[rep_mask_high_entropy[num_labeled :]]
        class_mask_u = torch.sum(
            prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3
        ).bool()

        # generate class mask for labeled data
        # label_l_mask = rep_mask_high_entropy[: num_labeled] * (label_l[:, i] == 0)
        # prob_i_classes = prob_indices_l[label_l_mask]
        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        class_mask = torch.cat(
            (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
        )

        negative_mask = rep_mask_high_entropy * class_mask

        keys = rep_teacher[negative_mask].detach()
        new_keys.append(
            dequeue_and_enqueue(
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )
        )

        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes.append(i)

    if (
            len(seg_num_list) <= 1
    ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep.sum()
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum()

    else:
        reco_loss = torch.tensor(0.0).cuda()
        seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
        valid_seg = len(seg_num_list)  # number of valid classes

        prototype = torch.zeros(
            (prob_indices_l.shape[-1], num_queries, 1, num_feat)
        ).cuda()

        for i in range(valid_seg):
            if (
                    len(seg_feat_low_entropy_list[i]) > 0
                    and memobank[valid_classes[i]][0].shape[0] > 0
            ):
                # select anchor pixel
                seg_low_entropy_idx = torch.randint(
                    len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                )
                anchor_feat = (
                    seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
                )
            else:
                # in some rare cases, all queries in the current query class are easy
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            # apply negative key sampling from memory bank (with no gradients)
            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().cuda()

                high_entropy_idx = torch.randint(
                    len(negative_feat), size=(num_queries * num_negatives,)
                )
                negative_feat = negative_feat[high_entropy_idx]
                negative_feat = negative_feat.reshape(
                    num_queries, num_negatives, num_feat
                )
                positive_feat = (
                    seg_proto[i]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(num_queries, 1, 1)
                    .cuda()
                )  # (num_queries, 1, num_feat)

                if momentum_prototype is not None:
                    if not (momentum_prototype == 0).all():
                        ema_decay = min(1 - 1 / i_iter, 0.999)
                        positive_feat = (
                                                1 - ema_decay
                                        ) * positive_feat + ema_decay * momentum_prototype[
                                            valid_classes[i]
                                        ]

                    prototype[valid_classes[i]] = positive_feat.clone()

                all_feat = torch.cat(
                    (positive_feat, negative_feat), dim=1
                )  # (num_queries, 1 + num_negative, num_feat)

            seg_logits = torch.cosine_similarity(
                anchor_feat.unsqueeze(1), all_feat, dim=2
            )

            reco_loss = reco_loss + F.cross_entropy(
                seg_logits / temp, torch.zeros(num_queries).long().cuda()
            )

        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg
        else:
            return prototype, new_keys, reco_loss / valid_seg


def get_criterion(cfg):
    cfg_criterion = cfg["criterion"]
    aux_weight = (
        cfg["net"]["aux_loss"]["loss_weight"]
        if cfg["net"].get("aux_loss", False)
        else 0
    )
    ignore_index = cfg["dataset"]["ignore_label"]
    if cfg_criterion["type"] == "ohem":
        criterion = CriterionOhem(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )
    else:
        criterion = Criterion(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )

    return criterion


def get_contra_loss(cfg, device='cpu'):
    cfg_contra = cfg["trainer"]["contrastive"]
    if 'pascal' in cfg['dataset']['type'] or 'VOC' in cfg['dataset']['type']:
        nclass = 21
        weak_list = [9, 18, 16, 11, 2, 4, 5, 20]
    elif 'cityscapes' in cfg['dataset']['type']:
        nclass = 19
        weak_list = [3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18]
    else:
        Exception('Wrong')
    temperature = cfg_contra['temperature']
    neg_num = cfg_contra['neg_num']
    memory_bank = cfg_contra['memory_bank']
    pixel_update_freq = cfg_contra['pixel_update_freq']
    memory_size = cfg_contra['memory_size']
    small_area = cfg_contra['small_area']
    feat_dim = cfg['net']['decoder']['kwargs']['inner_planes']
    max_positive = cfg_contra['max_positive']
    contra_loss = ContrastLoss(nclass, weak_list, temperature, neg_num, memory_bank,
                               pixel_update_freq, memory_size, small_area,
                               feat_dim, max_positive, device)
    return contra_loss


class Criterion(nn.Module):
    def __init__(self, aux_weight, ignore_index=255, use_weight=False):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            weights = torch.FloatTensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=weights
            )

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                    len(preds) == 2
                    and main_h == aux_h
                    and main_w == aux_w
                    and main_h == h
                    and main_w == w
            )
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(
                    main_pred, target
                )
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
        return loss


class CriterionOhem(nn.Module):
    def __init__(
            self,
            aux_weight,
            thresh=0.7,
            min_kept=100000,
            ignore_index=255,
            use_weight=False,
    ):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh, min_kept, use_weight
        )
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                    len(preds) == 2
                    and main_h == aux_h
                    and main_w == aux_w
                    and main_h == h
                    and main_w == w
            )

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (
                factor * factor
        )  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = (
            torch.from_numpy(input_label.reshape(target.size()))
            .long()
            .cuda(target.get_device())
        )

        return new_target

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
            self, ignore_index=255, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class ContrastLoss(nn.Module):
    def __init__(self, nclass, weak_list, temperature=0.1,
                 neg_num=64, memory_bank=True, pixel_update_freq=50,
                 memory_size=2000, small_area=True,
                 feat_dim=256, max_positive=False,
                 device='cpu',
                 methods=['cons', 'sep'],
                 ignore_label=255):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature
        self.max_positive = max_positive
        self.base_temperature = temperature
        self.ignore_label = ignore_label
        self.nclass = nclass
        self.n_steps = 0
        self.max_samples = int(neg_num * nclass)
        # 使用一个简单的小trick
        self.max_views = [neg_num // 4 for _ in range(nclass)]
        for index in weak_list:
            self.max_views[index] = neg_num
        self.small_area = small_area
        self.methods = methods
        if memory_bank:
            self.memory_bank = MemoryBank(nclass, memory_size, pixel_update_freq, feat_dim, device)
        else:
            self.memory_bank = None

    def _sampling(self, X, y_hat, y):
        batch_size = X.shape[0]
        y_hat = y_hat.contiguous().view(batch_size, -1)
        y = y.contiguous().view(batch_size, -1)
        X = X.contiguous().view(X.shape[0], -1, X.shape[-1])
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
        y_ = torch.empty(0).cuda()

        X_ptr = 0
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
                    all_indices = (this_y_hat == cls_id).nonzero()
                    num_all = all_indices.shape[0]
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
                Y = torch.ones(n_view).cuda() * cls_id
                y_ = torch.cat((y_, Y))
        return X_, y_

    def _contrastive(self, feats, labels):
        # 解释代码中的anchor_count和contrast_count
        # 原始代码来自于ContrastiveSeg库。 2021ICCV
        # 由于源代码在选取hard特征时，是选取了每个类别固定长度n_view个点的特征，其标签都被定义为一个数值。所以后续需要repeat。
        # 然而在保存各个类别的特征及其sampling时，得出的特征和类别信息一一对应，不需要repeat，所以count为1
        if feats is None:
            return None
        anchor_count = feats.shape[0]
        labels = labels.contiguous().view(-1, 1)

        anchor_feature = feats
        contrast_labels = labels
        contrast_feature = anchor_feature
        # contrast_count = anchor_count

        mask = torch.eq(labels, torch.transpose(contrast_labels, 0, 1)).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        # 跟自己的对比损失没必要计算
        # 如果是从队列里取出的特征就不需要从mask里面滤去了
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_count).view(-1, 1).cuda(), 0)
        # 这个mask是计算i与i+的关键部分。
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdims=True)
        # row, col = torch.where(neg_logits == 1)
        # labels_r = labels.repeat(64, 1)
        # labels_r[row[0]]
        # contrast_labels[col[0]]
        # 在这里打断点会发现memory bank中总有一些特征与当前的特征余弦相似度接近1 但是标签却不同
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

    def _contrastive_memory_bank(self, feats, labels, unlabeled_filter=False):
        # 解释代码中的anchor_count和contrast_count
        # 原始代码来自于ContrastiveSeg库。 2021ICCV
        # 由于源代码在选取hard特征时，是选取了每个类别固定长度n_view个点的特征，其标签都被定义为一个数值。所以后续需要repeat。
        # 然而在保存各个类别的特征及其sampling时，得出的特征和类别信息一一对应，不需要repeat，所以count为1
        if feats is None:
            return None
        anchor_count = feats.shape[0]
        device = feats.device
        labels = labels.contiguous().view(-1, 1)

        anchor_feature = feats

        memory_bank_use = self.memory_bank is not None and len(self.memory_bank) > 0
        if memory_bank_use:
            contrast_feature_que, contrast_labels_que = self.memory_bank.sample_queue_negative()
            contrast_feature_que, contrast_labels_que = contrast_feature_que.to(device), contrast_labels_que.to(device)
            contrast_labels_que = contrast_labels_que.contiguous().view(-1, 1)
            contrast_count_que = 1

        contrast_labels = labels
        contrast_feature = anchor_feature
        # contrast_count = anchor_count

        mask = torch.eq(labels, torch.transpose(contrast_labels, 0, 1)).float().cuda()
        # mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        # if not memory_bank_use:
        # 跟自己的对比损失没必要计算
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_count).view(-1, 1).cuda(), 0)
        # 这个mask是计算i与i+的关键部分。
        mask = mask * logits_mask
        if memory_bank_use:
            mask_que = torch.eq(labels, torch.transpose(contrast_labels_que, 0, 1)).float().cuda()
            neg_mask_que = 1 - mask_que
            mask = torch.cat([mask, mask_que], dim=1)
            neg_mask = torch.cat([neg_mask, neg_mask_que], dim=1)
            contrast_feature = torch.cat([contrast_feature, contrast_feature_que])

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if unlabeled_filter:
            parameter = unlabeled_filter.get_parameter()
            filtered_count = math.ceil((1-parameter) * anchor_count)
            pos_logit_pair_sum = (logits * mask).sum(1)
            _, indices = torch.topk(pos_logit_pair_sum, k=filtered_count, dim=0, largest=False)
            bool_index = torch.ones(logits.size(0), dtype=torch.bool)
            bool_index[indices] = False
            logits = logits[bool_index]
            neg_mask = neg_mask[bool_index]
            mask = mask[bool_index]
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdims=True)
        # 在这里打断点会发现memory bank中总有一些特征与当前的特征余弦相似度接近1 但是标签却不同
        # exp_logits = torch.exp(logits)
        # log_prob = logits - torch.log(exp_logits + neg_logits)
        # sum(1)是为了按照query的i来进行计算损失 现在计算出来的一列向量，每一列都是一个query的对比损失 最后计算平均得出损失函数
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if self.max_positive:
            logits_calmax = logits.masked_fill(~mask.bool(), -1 * 2 / self.temperature)
            logits_topn = logits_calmax.max(dim=1).values.unsqueeze(1)
            # logits_topn, _ = torch.topk(logits_calmax, k=5, dim=1)
            log_prob = logits_topn - torch.log(torch.exp(logits_topn) + neg_logits)
            loss = - (self.temperature / self.base_temperature) * log_prob
        else:
            logits = logits - torch.log(torch.exp(logits) + neg_logits)
            loss = - (self.temperature / self.base_temperature) * (mask * logits).sum(1) / mask.sum(1)
        nan_mask = torch.isnan(loss)
        loss = loss[~nan_mask]
        loss = loss.mean()

        return loss

    def forward(self, feats, labels, predict, unlabeled_filter=False):
        batchsize, _, h, w = feats.shape
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = predict.unsqueeze(1).float().clone()
        predict = torch.nn.functional.interpolate(predict, (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        predict = predict.squeeze(1).long()
        feats = feats.permute(0, 2, 3, 1)
        # memory_bank_use = self.memory_bank is not None and len(self.memory_bank) > 0
        feats_, labels_ = self._sampling(feats, labels, predict)
        if self.memory_bank is not None:
            loss = self._contrastive_memory_bank(feats_, labels_, unlabeled_filter)
        else:
            loss = self._contrastive(feats_, labels_)
        with torch.no_grad():
            if self.memory_bank is not None:
                method = 'sep' if 'sep' in self.methods else ''
                if 'cons' in self.methods:
                    sperate_ratio = self.memory_bank.consistence(feats_, labels_)
                    if self.memory_bank.best_ratio > sperate_ratio:
                        self.memory_bank.best_ratio = sperate_ratio
                sperate_ratio = self.memory_bank.dequeue_enqueue(feats, labels, self.small_area, method)
                if self.memory_bank.best_ratio > sperate_ratio:
                    self.memory_bank.best_ratio = sperate_ratio
        return loss


class MemoryBank:
    def __init__(self, class_num=19, memory_size=2000, pixel_update_freq=50,
                 feat_dim=128, device='cpu', split=1 / 2):
        super(MemoryBank, self).__init__()
        self.feat_dim = feat_dim
        self.device = device
        self.seg_queue = [torch.empty((feat_dim, 0)).to(self.device) for _ in range(class_num)]
        self.seg_queue_ptr = torch.zeros(class_num, dtype=torch.long).to(self.device)
        self.class_num = class_num
        self.memory_size = memory_size
        self.pixel_update_freq = pixel_update_freq
        self.best_ratio = 2 * self.class_num * self.class_num
        self.split = split

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

    def consistence(self, feats_, labels_):
        if self.__len__() > 0:
            feats_, labels_ = feats_.to(self.device), labels_.to(self.device)
            ahead_feat_que = [torch.empty((self.feat_dim, 0)).to(self.device) for _ in range(self.class_num)]
            classes = torch.unique(labels_)
            classes = [int(cls.item()) for cls in classes]
            none_classes = [i for i in range(self.class_num) if i not in classes]
            for cls in classes:
                length = self.seg_queue[cls].shape[1]
                index = (labels_ == cls)
                a = index.sum() / (index.sum() + length)
                feats_cls = feats_[index]
                feats_cls = feats_cls.view(-1, feats_cls.shape[-1])
                feat_m = feats_cls.mean(dim=0)
                ahead_feat_que[cls] = (1 - a) * self.seg_queue[cls] + a * feat_m.unsqueeze(1).repeat(1, length)
                ahead_feat_que[cls] = F.normalize(ahead_feat_que[cls], dim=0)
            for cls in none_classes:
                ahead_feat_que[cls] = self.seg_queue[cls]
            separate, ratio = separation(ahead_feat_que, self.seg_queue, self.class_num * (self.class_num - 1))
            if separate:
                self.seg_queue = ahead_feat_que
            with open('3.txt', 'a') as file:
                file.write('consistence {}:{}\n'.format(separate, ratio))
            return ratio
        else:
            return self.best_ratio

    def consistence_new(self, labels_):
        if self.__len__() > 0:
            labels_ = labels_.to(self.device)
            ahead_feat_que = [torch.empty((self.feat_dim, 0)).to(self.device) for _ in range(self.class_num)]
            classes = torch.unique(labels_)
            classes = [int(cls.item()) for cls in classes]
            none_classes = [i for i in range(self.class_num) if i not in classes]
            for cls in classes:
                length = self.seg_queue[cls].shape[1]
                a = self.split
                length_head = int(a * length)
                length_tail = length - length_head
                feats_cls_head_m = self.seg_queue[cls][:, :length_head].mean(dim=1)
                feats_cls_head_m = F.normalize(feats_cls_head_m, dim=0)
                coefficient_feat = torch.from_numpy(np.array([i for i in range(length_tail)])).cuda() / length_tail
                ahead_feat_que[cls] = self.seg_queue[cls]
                ahead_feat_que[cls][:, length_head:] = (1 - self.split) * (1 - coefficient_feat) * self.seg_queue[cls][:,
                                                                                                length_head:] + self.split * coefficient_feat * feats_cls_head_m.unsqueeze(
                    1).repeat(1, length_tail)
                ahead_feat_que[cls][:, length_head:] = F.normalize(ahead_feat_que[cls][:, length_head:], dim=0)
            for cls in none_classes:
                ahead_feat_que[cls] = self.seg_queue[cls]
            separate, ratio = separation(ahead_feat_que, self.seg_queue, self.class_num * (self.class_num - 1))
            if separate:
                self.seg_queue = ahead_feat_que
            with open('3.txt', 'a') as file:
                file.write('consistence {}:{}\n'.format(separate, ratio))
            return ratio
        else:
            return self.best_ratio

    def separation(self, feats_new):
        feats = self._mean_feature()
        feats = F.normalize(feats, dim=0)
        separate_ratio = (torch.matmul(torch.transpose(feats, 0, 1), feats)).sum() - self.class_num
        separate_ratio_new = (torch.matmul(torch.transpose(feats_new, 0, 1), feats_new)).sum() - self.class_num
        if separate_ratio_new < separate_ratio:
            return True, separate_ratio_new
        else:
            return False, separate_ratio

    def dequeue_enqueue(self, feats, labels, SMALL_AREA=True, methods='sep'):
        if 'sep' == methods:
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
            if 'sep' == methods:
                separate, ratio = separation(self.seg_queue, memory_queue,
                                             self.class_num * (self.class_num - 1),
                                             self.best_ratio)
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
            else:
                is_queue, ratio = separation(self.seg_queue, None,
                                             self.class_num * (self.class_num - 1),
                                             self.best_ratio)

            with open('3.txt', 'a') as file:
                file.write('separation {}:{}\n'.format(is_queue, ratio))
            return ratio
        else:
            return self.best_ratio

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
        # 这里的3*3的腐蚀核与上面的area>9有关的，不然会导致出错
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_dilate_np = cv2.erode(mask_dilate.int().cpu().numpy().astype('uint8'), kernel).astype('int32')
        mask_dilate = (torch.from_numpy(mask_dilate_np)).to(device).float()
        tmp = (mask_dilate.view(-1) == 1).nonzero()
        mask_list.append(tmp) if len(tmp) > 0 else None
        tmp = ((mask - mask_dilate).view(-1) == 1).nonzero()
        mask_list.append(tmp) if len(tmp) > 0 else None
        area = mask_dilate.sum()
    return mask_list


def separation(seg_queue_new, seg_queue_old=None, n=19 * 18, best_ratio=0):
    if seg_queue_old is not None:
        feats_new = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in seg_queue_new], dim=1)
        feats_new = F.normalize(feats_new, dim=0)
        separate_ratio_new = (torch.matmul(torch.transpose(feats_new, 0, 1), feats_new)).sum() + n
        if separate_ratio_new < best_ratio + 1:
            return True, separate_ratio_new
        feats_old = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in seg_queue_old], dim=1)
        feats_old = F.normalize(feats_old, dim=0)
        separate_ratio = (torch.matmul(torch.transpose(feats_old, 0, 1), feats_old)).sum() + n
        if separate_ratio_new < separate_ratio:
            return True, separate_ratio_new
        else:
            return False, separate_ratio
    else:
        feats_new = torch.cat([torch.mean(f, dim=1, keepdim=True) for f in seg_queue_new], dim=1)
        feats_new = F.normalize(feats_new, dim=0)
        separate_ratio_new = (torch.matmul(torch.transpose(feats_new, 0, 1), feats_new)).sum() + n
        return True, separate_ratio_new


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
            parameter = self.start_value + ((self.end_value - self.start_value) * math.log(1 + self.current_iter) / math.log(
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
