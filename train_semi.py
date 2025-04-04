import argparse
import copy
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from cs.dataset.augmentation import generate_unsup_data
from cs.dataset.builder import get_loader
from cs.models.model_helper import ModelBuilder
from cs.utils.dist_helper import setup_distributed
from cs.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
    get_contra_loss,
)
from cs.utils.lr_helper import get_optimizer, get_scheduler
from cs.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
    indicator_cal,
)
from eval import scale_crop_process

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument(
    "--config", type=str, default="experiments/cityscapes/372_pyr/ours/config.yaml"
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)


def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = 0, 1

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            osp.join(cfg["exp_path"], "log/events_seg/" + current_time)
        )
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)
    sup_loss_fn = get_criterion(cfg)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1
    total_iters = cfg_trainer["epochs"] * len(train_loader_sup)
    contra_loss_fn = None
    if cfg["trainer"].get("contrastive", False):
        if not cfg["trainer"]["contrastive"].get("method", "u2pl") == "u2pl":
            contra_loss_fn = get_contra_loss(cfg, device="cuda")

    # times = 10

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    # local_rank = 0
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=False,
    # )

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    # model_teacher = torch.nn.parallel.DistributedDataParallel(
    #     model_teacher,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=False,
    # )

    for p in model_teacher.parameters():
        p.requires_grad = False

    best_prec = 0
    last_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    if cfg["trainer"].get("contrastive", False):
        if cfg["trainer"]["contrastive"].get("method", "u2pl") == "u2pl":
            # build class-wise memory bank
            memobank = []
            queue_ptrlis = []
            queue_size = []
            for i in range(cfg["net"]["num_classes"]):
                memobank.append([torch.zeros(0, 256)])
                queue_size.append(30000)
                queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
            queue_size[0] = 50000

            # build prototype
            prototype = torch.zeros(
                (
                    cfg["net"]["num_classes"],
                    cfg["trainer"]["contrastive"]["num_queries"],
                    1,
                    256,
                )
            ).cuda()
    else:
        # build class-wise memory bank
        memobank = []
        queue_ptrlis = []
        queue_size = []
        for i in range(cfg["net"]["num_classes"]):
            memobank.append([torch.zeros(0, 256)])
            queue_size.append(30000)
            queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
        queue_size[0] = 50000

        # build prototype
        prototype = torch.zeros(
            (
                19,
                2000,
                1,
                256,
            )
        ).cuda()

    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        if cfg["trainer"].get("contrastive", False):
            if cfg["trainer"]["contrastive"].get("method", "u2pl") == "u2pl":
                # Training
                train(
                    model,
                    model_teacher,
                    optimizer,
                    lr_scheduler,
                    sup_loss_fn,
                    train_loader_sup,
                    train_loader_unsup,
                    epoch,
                    tb_logger,
                    logger,
                    memobank=memobank,
                    queue_ptrlis=queue_ptrlis,
                    queue_size=queue_size,
                )
            # 不使用U2PL
            else:
                train(
                    model,
                    model_teacher,
                    optimizer,
                    lr_scheduler,
                    sup_loss_fn,
                    train_loader_sup,
                    train_loader_unsup,
                    epoch,
                    tb_logger,
                    logger,
                    contra_loss=contra_loss_fn,
                )
        else:
            train(
                model,
                model_teacher,
                optimizer,
                lr_scheduler,
                sup_loss_fn,
                train_loader_sup,
                train_loader_unsup,
                epoch,
                tb_logger,
                logger,
                memobank=memobank,
                queue_ptrlis=queue_ptrlis,
                queue_size=queue_size,
            )

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if contra_loss_fn is not None:
            list1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 75, 100, 150, 200]
            if epoch + 1 in list1:
                features = contra_loss_fn.memory_bank.seg_queue
                gts = contra_loss_fn.memory_bank.gt_queue
                islabels = contra_loss_fn.memory_bank.islabel_queue
                torch.save(
                    features,
                    "./"
                    + cfg["saver"]["snapshot_dir"]
                    + "/"
                    + "features_{}".format(epoch + 1),
                )
                torch.save(
                    gts,
                    "./" + cfg["saver"]["snapshot_dir"] + "/" + "gt_{}".format(epoch + 1),
                )
                torch.save(
                    islabels,
                    "./"
                    + cfg["saver"]["snapshot_dir"]
                    + "/"
                    + "islabels_{}".format(epoch + 1),
                )
            sampling_num = contra_loss_fn.eval_bank.all
            accuracy = contra_loss_fn.eval_bank.indicator()
            tb_logger.add_scalar("sampling_num acc", sampling_num, epoch)
            tb_logger.add_scalar("sampling acc", accuracy, epoch)
            hard_num = contra_loss_fn.hard_samples.hard_num
            hard_right_num = contra_loss_fn.hard_samples.hard_right_num
            contra_loss_fn.hard_samples.clear()
            tb_logger.add_scalar("hard_num", hard_num, epoch)
            tb_logger.add_scalar("hard_right_num", hard_right_num, epoch)
        # Validation
        if cfg_trainer["eval_on"]:
            if rank == 0:
                logger.info("start evaluation")

            if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                prec = validate(model, val_loader, epoch, logger)
            else:
                prec = validate(model_teacher, val_loader, epoch, logger)

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "best_miou": best_prec,
                }
                if prec > best_prec:
                    best_prec = prec
                    torch.save(
                        state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                    )

                torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))

                logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                        best_prec * 100
                    )
                )
                tb_logger.add_scalar("mIoU val", prec, epoch)


def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    **kwargs,
):
    global prototype, cfg
    if cfg["trainer"].get("contrastive", False):
        if cfg["trainer"]["contrastive"].get("method", "u2pl") == "u2pl":
            memobank = kwargs["memobank"]
            queue_ptrlis = kwargs["queue_ptrlis"]
            queue_size = kwargs["queue_size"]
        else:
            contra_loss_fn = kwargs["contra_loss"]
    ema_decay_origin = cfg["net"]["ema_decay"]

    # loader_l.sampler.set_epoch(epoch)
    # loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    rank, world_size = 0, 1

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    con_losses_1 = AverageMeter(10)
    con_losses_2 = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(loader_l)):
        model.train()
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l, _ = loader_l_iter.next()
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        # image_u, _ = loader_u_iter.next()
        image_u, label_u, mask_u = (
            loader_u_iter.next()
        )  # 用来eval separation-driven sampling的准确率
        image_u = image_u.cuda()
        label_u = label_u.cuda()
        mask_u = mask_u.cuda()

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            contra_flag = "none"
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l)
            else:
                sup_loss = sup_loss_fn(pred, label_l)

            model_teacher.train()
            _ = model_teacher(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            # generate pseudo labels first
            model_teacher.eval()
            outs = model_teacher(image_u)
            pred_u_teacher, rep_u_teacher = outs["pred"], outs["rep"]
            pred_u_teacher = F.interpolate(
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(0, 1) < 1 and cfg["trainer"]["unsupervised"].get(
                "apply_aug", False
            ):
                (
                    image_u_aug,
                    logits_u_aug,
                    rep_u_t_aug,
                    label_u_aug,
                    mask_u_aug,
                    label_u_gt,
                ) = generate_unsup_data(
                    image_u,
                    logits_u_aug.clone(),
                    rep_u_teacher.clone(),
                    label_u_aug.clone(),
                    mask_u,
                    gt_label=label_u.clone(),
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                )
            else:
                mask_u_aug = mask_u
                rep_u_t_aug = rep_u_teacher
                image_u_aug = image_u

            # forward
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            outs = model(image_all)
            pred_all, rep_all = outs["pred"], outs["rep"]
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
            else:
                sup_loss = sup_loss_fn(pred_l_large, label_l.clone())

            # teacher forward
            if cfg["trainer"]["unsupervised"].get("method", "u2pl") == "u2pl":
                model_teacher.train()
                with torch.no_grad():
                    out_t = model_teacher(image_all)
                    pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                    prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                    prob_l_teacher, prob_u_teacher = (
                        prob_all_teacher[:num_labeled],
                        prob_all_teacher[num_labeled:],
                    )

                    pred_u_teacher = pred_all_teacher[num_labeled:]
                    pred_u_large_teacher = F.interpolate(
                        pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                    )

                # unsupervised loss using entropy threshold in U2PL
                drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
                percent_unreliable = (100 - drop_percent) * (
                    1 - epoch / cfg["trainer"]["epochs"]
                )
                drop_percent = 100 - percent_unreliable
                unsup_loss = compute_unsupervised_loss(
                    pred_u_large,
                    label_u_aug.clone(),
                    drop_percent,
                    pred_u_large_teacher.detach(),
                    mask=mask_u_aug,
                ) * cfg["trainer"]["unsupervised"].get("loss_weight", 1)

            contra_flag = "none"
            if cfg["trainer"].get("contrastive", False):
                # contrastive loss using unreliable pseudo labels
                if cfg["trainer"]["contrastive"].get("method", "u2pl") == "u2pl":
                    cfg_contra = cfg["trainer"]["contrastive"]
                    contra_flag = "{}:{}".format(
                        cfg_contra["low_rank"], cfg_contra["high_rank"]
                    )
                    alpha_t = cfg_contra["low_entropy_threshold"] * (
                        1 - epoch / cfg["trainer"]["epochs"]
                    )

                    with torch.no_grad():
                        prob = torch.softmax(pred_u_large_teacher, dim=1)
                        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                        low_thresh = np.percentile(
                            entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                        )
                        low_entropy_mask = (
                            entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                        )

                        high_thresh = np.percentile(
                            entropy[label_u_aug != 255].cpu().numpy().flatten(),
                            100 - alpha_t,
                        )
                        high_entropy_mask = (
                            entropy.ge(high_thresh).float()
                            * (label_u_aug != 255).bool()
                        )

                        low_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                low_entropy_mask.unsqueeze(1),
                            )
                        )

                        low_mask_all = F.interpolate(
                            low_mask_all, size=pred_all.shape[2:], mode="nearest"
                        )
                        # down sample

                        if cfg_contra.get("negative_high_entropy", True):
                            contra_flag += " high"
                            high_mask_all = torch.cat(
                                (
                                    (label_l.unsqueeze(1) != 255).float(),
                                    high_entropy_mask.unsqueeze(1),
                                )
                            )
                        else:
                            contra_flag += " low"
                            high_mask_all = torch.cat(
                                (
                                    (label_l.unsqueeze(1) != 255).float(),
                                    torch.ones(logits_u_aug.shape)
                                    .float()
                                    .unsqueeze(1)
                                    .cuda(),
                                ),
                            )
                        high_mask_all = F.interpolate(
                            high_mask_all, size=pred_all.shape[2:], mode="nearest"
                        )  # down sample

                        # down sample and concat
                        label_l_small = F.interpolate(
                            label_onehot(label_l, cfg["net"]["num_classes"]),
                            size=pred_all.shape[2:],
                            mode="nearest",
                        )
                        label_u_small = F.interpolate(
                            label_onehot(label_u_aug, cfg["net"]["num_classes"]),
                            size=pred_all.shape[2:],
                            mode="nearest",
                        )

                    if not cfg_contra.get("anchor_ema", False):
                        new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                        )
                    else:
                        prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                            prototype,
                        )

                    # dist.all_reduce(contra_loss)
                    contra_loss = (
                        contra_loss
                        / world_size
                        * cfg["trainer"]["contrastive"].get("loss_weight", 1)
                    )
                else:
                    weight = cfg["trainer"]["contrastive"].get("loss_weight", 1)
                    _, predict_label = torch.max(pred_all, dim=1)
                    contra_loss_outs = contra_loss_fn(
                        torch.nn.functional.normalize(rep_all[:num_labeled], dim=1),
                        torch.nn.functional.normalize(
                            rep_all_teacher[:num_labeled], dim=1
                        ),
                        torch.cat([label_l, label_u_aug])[:num_labeled],
                        predict_label[:num_labeled],
                        unlabeled=False,
                    )
                    contra_loss = contra_loss_outs["loss"] * weight
                    try:
                        contra_loss1 = contra_loss_outs["loss1"] * weight
                        contra_loss2 = contra_loss_outs["loss2"] * weight
                    except:
                        contra_loss1 = torch.Tensor(0)
                        contra_loss2 = torch.Tensor(0)
                    label_u_aug[mask_u_aug] = 255
                    contra_loss_outs = contra_loss_fn(
                        torch.nn.functional.normalize(rep_all[num_labeled:], dim=1),
                        torch.nn.functional.normalize(rep_u_t_aug, dim=1),
                        label_u_aug,
                        predict_label[num_labeled:],
                        unlabeled=True,
                        gtlabels=label_u_gt,
                    )
                    try:
                        contra_loss1 += contra_loss_outs["loss1"] * weight
                        contra_loss2 += contra_loss_outs["loss2"] * weight
                    except:
                        contra_loss1 = torch.Tensor(0)
                        contra_loss2 = torch.Tensor(0)
                    contra_loss += contra_loss_outs["loss"] * weight
            else:
                contra_loss = 0 * rep_all.sum()

        loss = sup_loss + unsup_loss + contra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                        + 1
                    ),
                    ema_decay_origin,
                )
                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        # dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        # dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = contra_loss.clone().detach()
        # dist.all_reduce(reduced_con_loss)
        con_losses.update(reduced_con_loss.item())

        # reduced_con_loss_1 = contra_loss1.clone().detach()
        # # dist.all_reduce(reduced_con_loss)
        # con_losses_1.update(reduced_con_loss_1.item())
        #
        # reduced_con_loss_2 = contra_loss2.clone().detach()
        # # dist.all_reduce(reduced_con_loss)
        # con_losses_2.update(reduced_con_loss_2.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}][{}] "
                "Iter [{}/{}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "LR {lr.val:.5f}".format(
                    cfg["dataset"]["n_sup"],
                    contra_flag,
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    data_time=data_times,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.avg, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.avg, i_iter)
            tb_logger.add_scalar("Con Loss", con_losses.avg, i_iter)
            tb_logger.add_scalar("Con Loss1", con_losses_1.avg, i_iter)
            tb_logger.add_scalar("Con Loss2", con_losses_2.avg, i_iter)


def validate(model, data_loader, epoch, logger, evalmodule=None):
    model.eval()
    # data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    # rank, world_size = dist.get_rank(), dist.get_world_size()
    rank, world_size = 0, 0

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        if "pascal" in cfg["dataset"]["type"]:
            with torch.no_grad():
                outs = model(images)
            # get the output produced by model_teacher
            output = outs["pred"]
            output = F.interpolate(
                output, labels.shape[1:], mode="bilinear", align_corners=True
            )
            output = output.data.max(1)[1].cpu().numpy()
        else:
            output = crop_forward(images, model)
            output = output.data.max(0)[1].unsqueeze(0).cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        # dist.all_reduce(reduced_intersection)
        # dist.all_reduce(reduced_union)
        # dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


def crop_forward(img, model):
    long_size = 2048
    classes = 19
    new_h = long_size
    new_w = long_size
    h, w = img.size()[-2:]
    if h > w:
        new_w = round(long_size / float(h) * w)
    else:
        new_h = round(long_size / float(w) * h)
    image_scale = F.interpolate(
        img, size=(new_h, new_w), mode="bilinear", align_corners=True
    )
    prediction = torch.zeros((classes, h, w), dtype=torch.float).cuda()
    prediction += scale_crop_process(model, image_scale, classes, 769, 769, h, w)
    return prediction


if __name__ == "__main__":
    main()
