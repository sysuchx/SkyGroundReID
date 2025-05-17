# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        # print('onnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn here')
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore = None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    # print('on111111111111111111111111111111111111111111 here')
                    if isinstance(score, list):
                        # print('score, list here')
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                        # for scor in score[0:]:
                        #     print(scor.shape,'scor[256, 1125]')
                        # print(len(score[0:]),'score[0:]  2')
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        # print('feat, list here')
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    if i2tscore != None:
                        # print('i2tscore here')
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        # loss = 0 + loss

                    # print(loss,'loss---')
                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion




def make_loss_air(cfg, num_classes):  # modified by gu
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        # print('onnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn here')
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:(air)", num_classes)
        def loss_fn_air(target,target_view,i2tscore=None):
            if i2tscore != None:
                indices = (target_view == 1).nonzero(as_tuple=True)[0]
                filtered_i2tscore = i2tscore[indices]
                filtered_target = target[indices]
                # I2TLOSS = xent(filtered_i2tscore, filtered_target)
                I2TLOSS = xent(i2tscore, target)
            return I2TLOSS
    return loss_fn_air





def make_loss_ground(cfg, num_classes):  # modified by gu
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        # print('onnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn here')
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:(ground)", num_classes)
        def loss_fn_ground(target,target_view,i2tscore=None):
            if i2tscore != None:
                indices = (target_view == 0).nonzero(as_tuple=True)[0]
                filtered_i2tscore = i2tscore[indices]
                filtered_target = target[indices]
                I2TLOSS = xent(filtered_i2tscore, filtered_target)
                # I2TLOSS = xent(i2tscore, target)
            return I2TLOSS

    return loss_fn_ground



def make_loss_stage2(cfg, num_classes):  # modified by gu
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        # print('onnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn here')
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:(air--------------------------------)", num_classes)
        def loss_fn_stage2(target,target_view,i2tscore=None):
            if i2tscore != None:
                # indices = (target_view == 1).nonzero(as_tuple=True)[0]
                # filtered_i2tscore = i2tscore[indices]
                # filtered_target = target[indices]
                # # I2TLOSS = xent(filtered_i2tscore, filtered_target)
                I2TLOSS = xent(i2tscore, target)
            return I2TLOSS
    return loss_fn_stage2


