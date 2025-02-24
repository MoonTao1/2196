import random
# import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# from torchvision import datasets
# from torchvision import transforms

import os
import json
import torch
import warnings
import argparse
import pickle
from networks.deeplabv3 import *
import logging
import time
from utils.mid_metrics import cc, sim, kldiv
from utils.options import parser
from utils.bulid_models import build_model
from utils.build_datasets import build_dataset
import matplotlib.pyplot as plt
model_root = 'models'
cuda = True
cudnn.benchmark = True
# lr = 1e-3
# batch_size = 16
# n_epoch = 1
warnings.simplefilter("ignore")

args = parser.parse_args()
# workers default = 4

ckpts = '/data9102/workspace/mwt/DANN-iter'

log_file = os.path.join(ckpts + "/train_log.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)
# load data
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    random.seed(2017)
    np.random.seed(2017)

    train_rainy_loader, valid_rainy_loader, test_rainy_loader = build_dataset(args=args)
    args.category = 'TrafficGaze'
    args.root = '/data/workspace/mwt/traffic_dataset/'
    train_loader, valid_loader, _ = build_dataset(args=args)
    print(len(train_loader))
    print(len(train_rainy_loader))
    model = build_model(args=args)

    params = model.parameters()
    cudnn.benchmark = True
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    # root = '/data/workspace/mwt/traffic_dataset/trafficframe/'
    # root1 = '/data/workspace/zcm/dataset/DrFixD-rainy/trafficframe/'
    # train_imgs = [json.loads(line) for line in open(root + 'train.json')]
    # train_rain_imgs = [json.loads(line) for line in open(root1 + 'train.json')]
    # valid_imgs = [json.loads(line) for line in open(root + 'valid.json')]
    # valid_rain_imgs = [json.loads(line) for line in open(root1 + 'valid.json')]


    # test_imgs = [json.loads(line) for line in open(root + 'n_test.json')]
    # test_rain_imgs = [json.loads(line) for line in open(root1 + 'n_test.json')]
    # print len(train_imgs),train_imgs[0]
    # print train_imgs
    # exit(0)
    best_loss = float('inf')
    file_name = os.path.join(ckpts, 'model_best.tar')
    print('-------------- New training session, LR = %.3f ----------------' %(args.lr))
    # print('-- length of training images = %d--length of valid images = %d--' % (len(train_imgs)+len(train_rain_imgs), len(valid_imgs)++len(valid_rain_imgs)))
    # train_loader = DataLoader(
    #     ImageList(root, train_imgs, for_train=True),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)
    #
    # train_rainy_loader = DataLoader(
    #     ImageList_r(root1, train_rain_imgs,for_train=True),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)
    #
    # valid_loader = DataLoader(
    #     ImageList(root, valid_imgs),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)
    #
    # valid_rainy_loader = DataLoader(
    #     ImageList_r(root1, valid_rain_imgs),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)

    # test_loader = DataLoader(
    #     ImageList(root, test_imgs),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)
    #
    # test_rainy_loader = DataLoader(
    #     ImageList_r(root1, test_rain_imgs),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)
    # load model
    # model = DeepLab(num_classes=1, backbone='mobilenet', output_stride=args.out_stride,
    #                             sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    # setup optimizer


    criterion = torch.nn.BCELoss()
    criterion_domain = torch.nn.NLLLoss()
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optim_dict'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_domain = criterion_domain.cuda()

    # for p in model.parameters():
    #     p.requires_grad = True

    # training
    for epoch in range(args.epochs):
        train_loss = train(train_loader,train_rainy_loader,model,criterion,criterion_domain,epoch,optimizer)
        valid_loss = validate(model, valid_rainy_loader, criterion,epoch)
        best_loss = min(valid_loss, best_loss)
        file_name_last = os.path.join(ckpts, 'model_epoch_%d.tar' % (epoch + 1,))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'valid_loss': valid_loss,
        }, file_name_last)

        if valid_loss == best_loss:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, file_name)
        print('Epoch: {:%d} Train loss {:%.4f} | Valid loss {:%.4f}' % (epoch+1, train_loss, valid_loss))

        # checkpoint = torch.load(file_name)
        # model.load_state_dict(checkpoint['state_dict'])
        # outputs, targets = predict(test_rainy_loader, model)
        #
        # np.savez(ckpts + 'test.npz', p=outputs, t=targets)
        # with open(ckpts + 'test.pkl', 'wb') as f:
        #     pickle.dump(test_imgs, f)

        # accu_s = test(source_dataset_name)
        # print('Accuracy of the %s dataset: %f' % ('mnist', accu_s))
        # accu_t = test(target_dataset_name)
        # print('Accuracy of the %s dataset: %f\n' % ('mnist_m', accu_t))
        # if accu_t > best_accu_t:
        #     best_accu_s = accu_s
        #     best_accu_t = accu_t
        #     torch.save(my_net, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))

    print('============ Summary ============= \n')
# print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
# print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
    print('Corresponding model was save in ' + model_root + '/model_epoch_best.tar')


def validate(model, valid_loader, criterion,epoch):
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    alpha = 0
    start = time.time()
    metrics = [0, 0, 0]
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            from torchvision.transforms import Resize
            torch_resize = Resize([256, 256])  # 定义Resize类对象
            target = torch_resize(target)
            input = input.cuda()
            target = target.cuda()
            # compute output
            output,_ = model(input,alpha)

            loss = criterion(output, target)

            # measure accuracy and record loss

            losses.update(loss.data, target.size(0))
            # valid metrics printing
            output = output.squeeze(1)
            target = target.squeeze(1)
            metrics[0] = metrics[0] + cc(output, target)
            metrics[1] = metrics[1] + sim(output, target)
            metrics[2] = metrics[2] + kldiv(output, target)

            msg = 'epoch: {:03d} Validating Iter {:03d} Loss {:.6f} || CC {:4f}  SIM {:4f}  KLD {:4f} in {:.3f}s'.format(epoch+1,i + 1,
                                                                                                           losses.avg,
                                                                                                           metrics[
                                                                                                               0] / (
                                                                                                                   i + 1),
                                                                                                           metrics[
                                                                                                               1] / (
                                                                                                                   i + 1),
                                                                                                           metrics[
                                                                                                               2] / (
                                                                                                                   i + 1),
                                                                                                           time.time() - start)
            # print(msg)
            # logging.info(msg)
            start = time.time()

            del input, target, output
            # gc.collect()

            interval = 5
            if (i + 1) % interval == 0:
                logging.info(msg)

    model.train()

    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, train_rainy_loader,model, criterion,criterion_domain,epoch,optimizer):
    erres = AverageMeter()
    model.train()
    # 开启训练模式
    len_dataloader = min(len(train_loader), len(train_rainy_loader))
    data_source_iter = iter(train_loader)
    data_target_iter = iter(train_rainy_loader)

    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.__next__()
        s_img, s_label = data_source

        batch_size = len(s_label)
        # 传入源域数据
        domain_label = torch.zeros(batch_size).long()
        model.zero_grad()
        # 梯度清零
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()

        class_output, domain_output = model(input_data=s_img, alpha=alpha)
        err_s_label = criterion(class_output, s_label)
        err_s_domain = criterion_domain(domain_output, domain_label)
        # 前向传播确定损失
        data_target = data_target_iter.__next__()
        t_img, _ = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()
        # 加载目标域损失
        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        _, domain_output = model(input_data=t_img, alpha=alpha)
        err_t_domain = criterion_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        erres.update(err.item(), s_label.size(0))
        # 计算损失值并记录
        err.backward()
        optimizer.step()
        # 反向传播并进行优化
        sys.stdout.write('\r train epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch+1, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()

    print('\n')
    return erres.avg
#
# def predict(test_loader, model):
#     model.eval()
#     len_dataloader = len(test_loader)
#     targets = []
#     outputs = []
#     data_source_iter = iter(test_loader)
#     for i in range(len_dataloader):
#         # ----------改变target的size--------------------
#         # print('validating-第{}次,input_size:{}'.format(i, input.shape))
#         # print('validating-第{}次,target_size:{}'.format(i, target.shape))
#         # print(target)
#         data_source = data_source_iter.__next__()
#         input, target = data_source
#         from torchvision.transforms import Resize
#         torch_resize = Resize([192, 320])  # 定义Resize类对象
#         target= torch_resize(target)
#         # ---------------------------------------------
#
#         targets.append(target.numpy().squeeze(1))
#
#         input = input.cuda()
#
#         # compute output
#         output = model(input)
#         outputs.append(output.data.cpu().numpy().squeeze(1))
#
#     targets = np.concatenate(targets)
#     outputs = np.concatenate(outputs)
#     return outputs, targets


if __name__ == '__main__':
    main()