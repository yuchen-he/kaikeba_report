# This file includes the config definitions, which will be called by detector_myself.py
from __future__ import print_function
import os
import logging

import torch


def my_args_parser(argparser):
    # dataset configs
    argparser.add_argument('--gen_data', type=str, default='stage3',
                           help='stage1/stage1_inspect/stage3/stage3_inspect')
    argparser.add_argument('--train_set_path', type=str, default='./data/train.txt',
                           help='the relative path to train dataset')
    argparser.add_argument('--test_set_path', type=str, default='./data/test.txt',
                           help='the relative path to test dataset')
    argparser.add_argument('--train_set_path_stage3', type=str, default='./data/train_stage3.txt',
                           help='the relative path to train dataset')
    argparser.add_argument('--test_set_path_stage3', type=str, default='./data/test_stage3.txt',
                           help='the relative path to test dataset')
    argparser.add_argument('--input_size', type=int, default=112,
                           help='input image size for training (default: 112)')
    # data augmentation configs
    argparser.add_argument('--no_normalize', action='store_true', default=False,
                           help='not to do normalize for img and gt_landmarks')
    argparser.add_argument('--rotate_angle', type=int, default=30, metavar='N',
                           help='rotation angle for data augmentation (default: 30)')
    # train configs
    argparser.add_argument('--stage', type=str, default='3',
                           help='1/3')
    argparser.add_argument('--model', type=str, default='OrigNet',  # Resnet18/OrigNet
                           help='backbone model')
    argparser.add_argument('--batch_size', type=int, default=64, metavar='N',
                           help='input batch size for training (default: 64)')
    argparser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                           help='input batch size for testing (default: 64)')
    argparser.add_argument('--epochs', type=int, default=100, metavar='N',
                           help='number of epochs to train (default: 100)')
    argparser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                           help='learning rate (default: 0.001)')
    argparser.add_argument('--optimizer', type=str, default='sgd',
                           help='sgd/adam')
    argparser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                           help='SGD momentum (default: 0.5)')
    argparser.add_argument('--no_cuda', action='store_true', default=False,
                           help='disables CUDA training')
    argparser.add_argument('--seed', type=int, default=1, metavar='S',
                           help='random seed (default: 1)')
    argparser.add_argument('--log-interval', type=int, default=5, metavar='N',
                           help='how many batches to wait before logging training status')
    argparser.add_argument('--save_model', action='store_true', default=True,
                           help='save the current Model')
    argparser.add_argument('--save_directory', type=str, default='trained_models',
                           help='learnt models are saving here')
    argparser.add_argument('--phase', type=str, default='Train',  # Train/train, Predict/predict, Finetune/finetune
                           help='training, predicting or finetuning')
    # test/predict configs
    argparser.add_argument('--trained_model_path', type=str, default='trained_models/model_best.pth',
                           help='load trained base_model here')
    argparser.add_argument('--img_path', type=str, default='data/test_imgs',
                           help='image path for predicting')
    # finetune configs
    argparser.add_argument('--is_finetune', action='store_true', default=False,
                           help='finetune')
    argparser.add_argument('--finetune_lr', type=float, default=0.0005, metavar='LR',
                           help='learning rate (default: 0.0005)')
    argparser.add_argument('--fine_tune_path', type=str, default='trained_models/model_best.pth',
                           help='load finetune base_model here')
    args = argparser.parse_args()
    return args


def create_logger(name, file_name, level=logging.INFO):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s] [%(filename)15s] [%(levelname)8s] %(message)s')

    # for save logs in file
    fh = logging.FileHandler(file_name, 'w')
    fh.setFormatter(formatter)
    # for show logs in terminal
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    # logger.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def load_state(path, model):
    if os.path.isfile(path):
        print(f"==========> Loading model from {path}")
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        raise IOError("The trained model path is not exist!")



