# This file includes the config definitions, which will be called by detector_myself.py
import argparse
import json


def my_args_parser(argparser):
    # dataset configs
    argparser.add_argument('--train_set_path', type=str, default='./data/train.txt',
                           help='the relative path to train dataset')
    argparser.add_argument('--test_set_path', type=str, default='./data/test.txt',
                           help='the relative path to test dataset')
    argparser.add_argument('--input_size', type=int, default=112,
                           help='input image size for training (default: 112)')
    # train configs
    argparser.add_argument('--batch-size', type=int, default=64, metavar='N',
                           help='input batch size for training (default: 64)')
    argparser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                           help='input batch size for testing (default: 64)')
    argparser.add_argument('--epochs', type=int, default=100, metavar='N',
                           help='number of epochs to train (default: 100)')
    argparser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                           help='learning rate (default: 0.001)')
    argparser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                           help='SGD momentum (default: 0.5)')
    argparser.add_argument('--no-cuda', action='store_true', default=False,
                           help='disables CUDA training')
    argparser.add_argument('--seed', type=int, default=1, metavar='S',
                           help='random seed (default: 1)')
    argparser.add_argument('--log-interval', type=int, default=20, metavar='N',
                           help='how many batches to wait before logging training status')
    argparser.add_argument('--save-model', action='store_true', default=True,
                           help='save the current Model')
    argparser.add_argument('--save-directory', type=str, default='trained_models',
                           help='learnt models are saving here')
    argparser.add_argument('--phase', type=str, default='Train',  # Train/train, Predict/predict, Finetune/finetune
                           help='training, predicting or finetuning')
    args = argparser.parse_args()
    return args
