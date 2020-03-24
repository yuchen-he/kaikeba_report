from __future__ import print_function
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

try:
    from tensorboardX import SummaryWriter

    USE_TENSORBOARDX = True
    print('Use tensorboardX!')
except:
    USE_TENSORBOARDX = False
    print('TensorboardX is not exsit')

from networks.network_stage3 import Net_face
import my_dataloader_stage3
from utils import utils
from my_predictor import predictor
import pdb


def train(args, train_loader, valid_loader, model, my_criterion, device, logger, tf_writer):
    num_epoch = args.epochs
    save_path = args.save_directory
    loss_best = 1e10
    # total_step = int(len(train_loader) / args.batch_size * num_epoch)
    model.to(device)

    # finetune
    if args.is_finetune:
        utils.load_state(args.fine_tune_path, model)
        args.lr = args.finetune_lr
        finetune_layers = ['fc_1', 'fc_2']  # change to config file later
        for p in model.parameters():
            p.requires_grad = False

        logger.info("The following layers are going to be finetune:")
        for finetune_layer_name in finetune_layers:
            logger.info(finetune_layer_name)
            # also can use "for name,param in model_ft.named_parameters():"
            # if name == finetune_layer_name: param.requires_grad = False
            for p in getattr(model, finetune_layer_name).parameters():
                p.requires_grad = True
        # pdb.set_trace()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        b1 = 0.9  # for momentom
        b2 = 0.999  # for RMSProp
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr * 5, betas=(b1, b2))

    logger.info("Please check your configs:\n{}".format(args))
    for epoch_id in range(num_epoch):
        train_losses = 0
        valid_losses_kp = 0
        valid_losses_face = 0
        train_step = 0
        val_step = 0

        ########################
        ### training process ###
        ########################
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            train_step += 1

            # ground truth
            input_img = batch['image'].to(device)  # input_img shape: N x C x H x W
            face_gt = batch['is_face'].to(device)
            landmarks_gt = batch['landmarks'].to(device)
            # print(f"input_size: {input_img.size()},\n gt_size: {landmarks_gt.size()}")

            # zero grad
            optimizer.zero_grad()

            # process the batch
            kp_output, face_output = model(input_img)  # torch.Size([N, 42])

            # calculate loss and back propagation
            # 1. face_loss
            face_gt = torch.squeeze(face_gt)
            face_loss = my_criterion['face'](face_output, face_gt)
            # 2. kp_loss
            is_face_mask = torch.cuda.ByteTensor(landmarks_gt.size()).bool()
            is_face_mask.zero_()
            face_gt_list = face_gt.cpu().numpy().tolist()
            for i in range(len(face_gt_list)):
                if face_gt[i] == 1:
                    is_face_mask[i] = 1
            kp_pred = kp_output[is_face_mask].view(-1, 42)
            kp_gt = landmarks_gt[is_face_mask].view(-1, 42)
            kp_loss = my_criterion['landmark'](kp_pred, kp_gt)

            train_loss = kp_loss + 2 * face_loss
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()

            # save logs
            if batch_idx % args.log_interval == 0:
                logger.info("Epoch:{} [{}/{} ({:.0f}%)]\t kp_loss: {:.6f} face_loss: {:.6f}".format(
                    epoch_id,
                    batch_idx * len(input_img),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    kp_loss.item(),
                    face_loss.item()
                ))
        train_losses /= 1.0 * train_step

        ########################
        ## validation process ##
        ########################
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                val_step += 1
                input_img = batch['image'].to(device)  # input_img shape: N x C x H x W
                face_gt = batch['is_face'].to(device)
                landmarks_gt = batch['landmarks'].to(device)
                kp_output, face_output = model(input_img)
                # 1. face_loss
                face_gt = torch.squeeze(face_gt)
                face_loss = my_criterion['face'](face_output, face_gt)
                # 2. kp_loss
                is_face_mask = torch.cuda.ByteTensor(landmarks_gt.size()).bool()
                is_face_mask.zero_()
                face_gt_list = face_gt.cpu().numpy().tolist()
                for i in range(len(face_gt_list)):
                    if face_gt[i] == 1:
                        is_face_mask[i] = 1
                kp_pred = kp_output[is_face_mask].view(-1, 42)
                kp_gt = landmarks_gt[is_face_mask].view(-1, 42)
                kp_loss = my_criterion['landmark'](kp_pred, kp_gt)

                valid_losses_kp += kp_loss.item()
                valid_losses_face += face_loss.item()

            valid_losses_kp /= val_step * 1.0
            valid_losses_face /= val_step * 1.0
            # save logs per epoch
            logger.info('Landmark Evaluation loss: {:.6f}'.format(valid_losses_kp))
            logger.info('face Evaluation loss: {:.6f}'.format(valid_losses_face))
            logger.info('====================================================')

        # save model only when loss_best
        if (valid_losses_kp + valid_losses_face) <= loss_best:
            loss_best = valid_losses_kp + valid_losses_face
            if args.save_model:
                if args.is_finetune:
                    saved_model_name = os.path.join(save_path, 'model_best_finetuned.pth')
                else:
                    saved_model_name = os.path.join(save_path, 'model_best.pth')
                torch.save(model.state_dict(), saved_model_name)

        # save tensorboardX info //todo: count true/false samples, then get accurage for each
        # tf_writer.add_scalar("train_loss", train_losses, epoch_id)
        # tf_writer.add_scalar("val_loss_landmark", valid_losses_kp, epoch_id)
        # tf_writer.add_scalar("val_loss_face", valid_losses_face, epoch_id)

    return train_losses, loss_best


def test(args, valid_loader, model, my_criterion, device):
    val_step = 0
    valid_losses = 0
    batch_time = 0

    utils.load_state(args.trained_model_path, model)
    model.to(device)
    # same with eval code
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            val_step += 1
            torch.cuda.synchronize()
            start_time = time.time()

            # process image
            input_img = batch['image'].to(device)  # input_img shape: N x C x H x W
            landmarks_gt = batch['landmarks'].to(device)
            output = model(input_img)
            valid_loss = my_criterion(output, landmarks_gt)

            # loss calculation
            valid_losses += valid_loss.item()

            # batch time calculate
            torch.cuda.synchronize()
            batch_process_t = time.time() - start_time
            batch_time += batch_process_t

        valid_losses /= val_step * 1.0
        batch_time /= val_step * 1.0
        print('Evaluation loss: {:.6f}\nBatch {} process time: {}'.format(
            valid_losses, args.test_batch_size, batch_time))


def main(args):
    torch.manual_seed(args.seed)

    # step1: set single gpu
    print('===> Step1: Set Gpu')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('step1 done!')

    # step2: set dataset
    print('===> Step2: Loading Datasets')
    train_set, valid_set = my_dataloader_stage3.dataloader(args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.test_batch_size, shuffle=False)
    print('step2 done!')

    # step3: set networks
    print('===> Step3: Building Model')
    model = Net_face()  # for later stage, change it to be selected by config
    # model = resnet18(pretrained=False)
    print('step3 done!')

    # step4: set optimizer
    # //todo: add lr_warmup function later to compare the results
    print('===> Step4: Set Optimizer')
    criterion_kp = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    my_criterion = {'landmark': criterion_kp, 'face': criterion_cls}
    # my_criterion = nn.SmoothL1Loss()
    print('step4 done!')

    # step5: train or test or demo
    args.save_directory = os.path.join(args.save_directory, "FaceNet_NoNormalize_" + str(args.no_normalize) +
                                       "_" + args.optimizer + "_finetune_" + str(args.is_finetune))
    if not os.path.exists(args.save_directory):
        os.system("mkdir -p {}".format(args.save_directory))
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Step5: Start Training')
        logger = utils.create_logger('my_logger', os.path.join(args.save_directory, 'log.txt'))
        if USE_TENSORBOARDX:
            tf_writer = SummaryWriter(os.path.join(args.save_directory, 'tensorboard'))
        train_losses, valid_losses = \
            train(args, train_loader, valid_loader, model, my_criterion, device, logger, tf_writer)
        print('====================================================')
        tf_writer.close()
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Step5: Test')
        test(args, valid_loader, model, my_criterion, device)
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Step5: Finetune')
        logger = utils.create_logger('my_logger', os.path.join(args.save_directory, 'log.txt'))
        if USE_TENSORBOARDX:
            tf_writer = SummaryWriter(os.path.join(args.save_directory, 'tensorboard'))
        train_losses, valid_losses = \
            train(args, train_loader, valid_loader, model, my_criterion, device, logger, tf_writer)
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Step5: Predict')
        predictor(args, model, device)
    print('step5: {} done!'.format(args.phase))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Face_Landmarks_Detection")
    args = utils.my_args_parser(argparser)
    main(args)
