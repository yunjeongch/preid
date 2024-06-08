import logging
import os
import random
import time
import torch
import torch.nn as nn
from model.make_model import make_model
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP_with_pose
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from loss.inferability import periodic_distribution_weights

def part_attention_vit_do_train_with_amp(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank,
             patch_centers = None,
             pc_criterion= None,
             pose_dict = None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("PAT.train")
    logger.info('start training')
    tb_path = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print("saving tblog to {}".format(tb_path))
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    total_loss_meter = AverageMeter()
    reid_loss_meter = AverageMeter()
    pc_loss_meter = AverageMeter()
    # ds_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512)
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    # train
    if cfg.MODEL.PC_LOSS:
        print('initialize the centers')
        model.train()
        for i, informations in enumerate(train_loader):
            # measure data loading time
            with torch.no_grad():
                #input = input.cuda(non_blocking=True)
                input = informations['images'].cuda(non_blocking=True)
                vid = informations['targets']
                camid = informations['camid']
                path = informations['img_path']
                #input = input.view(-1, input.size(2), input.size(3), input.size(4))

                # compute output
                _, _, layerwise_feat_list = model(input)
                patch_centers.get_soft_label(path, layerwise_feat_list[-1], vid=vid, camid=camid)
        print('initialization done')
    
    best_mAP = 0.0
    best_index = 1
    stop_training = False
    for epoch in range(1, epochs + 1):
        if stop_training:
            break
        start_time = time.time()
        total_loss_meter.reset()
        reid_loss_meter.reset()
        acc_meter.reset()
        pc_loss_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, informations in enumerate(train_loader):
            img = informations['images']
            vid = informations['targets']
            camid = informations['camid']
            img_path = informations['img_path']
            # t_domains = informations['others']['domains']

            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = camid.to(device)
            # t_domains = t_domains.to(device)

            model.to(device)
            with amp.autocast(enabled=True):
                score, layerwise_global_feat, layerwise_feat_list = model(img)
                if cfg.INFERABILITY.TRIPLET:
                    _3d_dir = torch.tensor([pose_dict[q] for q  in img_path]).to(device)
                
                ############## patch learning ######################
                patch_agent, position = patch_centers.get_soft_label(img_path, layerwise_feat_list[-1], vid=vid, camid=camid)   # patch_agent: [3, N, emb_dim], position: [N,]
                l_ploss = cfg.MODEL.PC_LR
                if cfg.MODEL.PC_LOSS:
                    feat = torch.stack(layerwise_feat_list[-1], dim=0) # stack last layer's emb -> [3, B, emb_dim]
                    feat = feat[:,::1,:]
                    '''
                    loss1: clustering loss(for patch centers)
                    '''
                    ploss, all_posvid = pc_criterion(feat, patch_agent, position, patch_centers, vid=target, camid=target_cam)
                    '''
                    loss2: reid-specific loss
                    (ID + Triplet loss)
                    '''
                    
                    reid_loss = loss_fn(score, layerwise_global_feat[-1], target, all_posvid=all_posvid, soft_label=cfg.MODEL.SOFT_LABEL, soft_weight=cfg.MODEL.SOFT_WEIGHT, soft_lambda=cfg.MODEL.SOFT_LAMBDA, _3d_dir = _3d_dir)
                    if np.isnan(reid_loss.item()):
                        stop_training = True
                        break
                else:
                    ploss = torch.tensor([0.]).cuda()
                    reid_loss = loss_fn(score, layerwise_global_feat[-1], target, soft_label=cfg.MODEL.SOFT_LABEL)
                
                total_loss = reid_loss + l_ploss*ploss

            scaler.scale(total_loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # score = scores[-1]
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            total_loss_meter.update(total_loss.item(), img.shape[0])
            reid_loss_meter.update(reid_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            pc_loss_meter.update(ploss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] total_loss: {:.3f}, reid_loss: {:.3f}, pc_loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader), total_loss_meter.avg,
                reid_loss_meter.avg, pc_loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/reid_loss', reid_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar("train/pc_loss", pc_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
        if stop_training:
            print('the loss became nan.... terminating!')
            break

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
        
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img)
                            evaluator.update((feat, vid, camid, img_path))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                cmc, mAP = do_inference(cfg, model, val_loader, num_query)
                tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
                tbWriter.add_scalar('val/mAP', mAP, epoch)

        if epoch % checkpoint_period == 0:
            if best_mAP < mAP:
                best_mAP = mAP
                best_index = epoch
                logger.info("=====best epoch: {}=====".format(best_index))
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        torch.cuda.empty_cache()

    # final evaluation
    load_path = os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(best_index))
    eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0, camera_num=None, view_num=None)
    eval_model.load_param(load_path)
    print('load weights from {}_{}.pth'.format(cfg.MODEL.NAME, best_index))
    for testname in cfg.DATASETS.TEST:
        if 'ALL' in testname:
            testname = 'DG_' + testname.split('_')[1]
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference(cfg, eval_model, val_loader, num_query)
    
    # remove useless path files
    del_list = os.listdir(log_path)
    for fname in del_list:
        if '.pth' in fname:
            os.remove(os.path.join(log_path, fname))
            print('removing {}. '.format(os.path.join(log_path, fname)))
    # save final checkpoint
    print('saving final checkpoint.\nDo not interrupt the program!!!')
    torch.save(eval_model.state_dict(), os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(max(epoch, 60))))
    print('done!')

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 pose_dict = None):
    device = "cuda"
    logger = logging.getLogger("PAT.test")
    logger.info("Enter inferencing")

    # if not (pose_dict is None):
    #     pose_ranker = R1_mAP_with_pose(num_query, max_rank = 50, feat_norm = cfg.TEST.FEAT_NORM)
    #     pose_ranker.reset()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    t0 = time.time()
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        batch_size, _, width, _ = img.shape
        # widths = torch.tensor([width] * batch_size).unsqueeze(dim = -1).unsqueeze(dim = -1).to(device)
        # domains = informations['others']['domains']
        with torch.no_grad():
            img = img.to(device)
            # camids = camids.to(device)
            feat = model(img)
            # if not (pose_dict is None):
            #     poses = list(pose_model(imgpath))
            #     keypoints = torch.tensor([poses[sample]['predictions'][0][0]['keypoints'] for sample in range(len(imgpath))]).to(device)
            #     directions = calculate_directions(keypoints, widths)
            #     pose_ranker.update((feat, pid, camids, imgpath, directions))
            evaluator.update((feat, pid, camids, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("total inference time: {:.2f}".format(time.time() - t0))
    # if not (pose_dict is None):
    #     pose_ranker.relative_rank_of_GT(save_dir = cfg.DATASETS.TRAIN[0] + '_' + '_' + cfg.LOG_NAME.replace('/', '') + cfg.DATASETS.TEST[0])
    return cmc, mAP

def do_inference_with_save(cfg,
                 model,
                 val_loader,
                 num_query,
                 save_dir
                ):
    device = "cuda"
    logger = logging.getLogger("PAT.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    t0 = time.time()
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        # domains = informations['others']['domains']
        with torch.no_grad():
            img = img.to(device)
            # camids = camids.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camids, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute(save_dir)
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP