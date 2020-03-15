from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from networks.network_stage1 import Net_Original
import my_dataloader
from utils.parse_config import my_args_parser
import pdb


def train(args, train_loader, valid_loader, model, my_criterion, optimizer, device):
    num_epoch = args.epochs
    save_path = args.save_directory
    loss_best = 1e10
    total_step = int(len(train_loader) / args.batch_size * num_epoch)
    curr_step = 0

    for epoch_id in range(num_epoch):
        # train_losses = 0
        valid_losses = 0

        ########################
        ### training process ###
        ########################

        model.train()
        for batch_idx, batch in enumerate(train_loader):

            # ground truth
            input_img = batch['image'].to(device)  # input_img shape: N x C x H x W
            landmarks_gt = batch['landmarks'].to(device)
            # print(f"input_size: {input_img.size()},\n gt_size: {landmarks_gt.size()}")

            # zero grad
            optimizer.zero_grad()

            # process the batch
            output = model(input_img)     # torch.Size([N, 42])
            # print(f"output_size: {output.size()}")

            # calculate loss and back propagation
            train_losses = my_criterion(output, landmarks_gt)
            train_losses.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t train_loss: {:.6f}'.format(
                    epoch_id,
                    batch_idx * len(input_img),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    train_losses.item()
                )
                )

        ########################
        ## validation process ##
        ########################

        model.eval()
        with torch.no_grad():
            val_step = 0

            for batch_idx, batch in enumerate(valid_loader):
                val_step += 1
                input_img = batch['image'].to(device)  # input_img shape: N x C x H x W
                landmarks_gt = batch['landmarks'].to(device)
                output = model(input_img)
                valid_loss = my_criterion(output, landmarks_gt)
                valid_losses += valid_loss.item()

            valid_losses /= val_step * 1.0
            print('Valid: pts_loss: {:.6f}'.format(valid_losses))
            print('====================================================')

        # save model only when loss_best
        if valid_losses <= loss_best:
            loss_best = valid_losses
            if args.save_model:
                saved_model_name = os.path.join(save_path, 'model_best.pth')
                torch.save(model.state_dict(), saved_model_name)

    return train_losses, loss_best


# test(model, valid_loader)
# finetune()

def main(args):
    torch.manual_seed(args.seed)

    # step1: set single gpu
    print('===> Step1: Set Gpu')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('step1 done!')

    # step2: set dataset
    print('===> Step2: Loading Datasets')
    train_set, valid_set = my_dataloader.dataloader(args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.test_batch_size, shuffle=False)
    print('step2 done!')

    # step3: set networks
    print('===> Step3: Building Model')
    model = Net_Original().to(device)  # for later stage, change it to be selected by config
    print('step3 done!')

    # step4: set optimizer
    # //todo: add lr_warmup function later to compare the results
    print('===> Step4: Set Optimizer')
    my_criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print('step4 done!')

    # step5: train or test or demo
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Step5: Start Training')
        train_losses, valid_losses = \
            train(args, train_loader, valid_loader, model, my_criterion, optimizer, device)
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Step5: Test')
        # how to do test?
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Step5: Finetune')
        # how to do finetune?
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Step5: Predict')
        # how to do predict?
    print('step5: {} done!'.format(args.phase))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Face_Landmarks_Detection")
    args = my_args_parser(argparser)
    main(args)
