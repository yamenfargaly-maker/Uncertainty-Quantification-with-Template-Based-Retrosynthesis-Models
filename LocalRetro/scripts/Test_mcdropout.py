from argparse import ArgumentParser

import torch
import torch.nn as nn

from utils import init_featurizer, mkdir_p, get_configure, load_model, load_dataloader, predict
from mc_dropout_utils import write_edits_mc_dropout

def main(args):
    model_name = 'LocalRetro_%s.pth' % args['dataset']
    args['model_path'] = '../models/%s' % model_name
    args['config_path'] = '../data/configs/%s' % args['config']
    args['data_dir'] = '../data/%s' % args['dataset']
    args['result_path'] = '../outputs/raw_prediction/%s' % model_name.replace('.pth', '_mcdropout.txt')
    args['mc_uq_path'] = '../outputs/raw_prediction/%s' % model_name.replace('.pth', '_mcdropout_uq.txt')
    mkdir_p('../outputs')
    mkdir_p('../outputs/raw_prediction')

    args = init_featurizer(args)
    model = load_model(args)
    test_loader = load_dataloader(args)
    write_edits_mc_dropout(args, model, test_loader, n_passes=args['n_passes'])
    return

if __name__ == '__main__':
    parser = ArgumentParser('LocalRetro MC-dropout testing arguments')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-c', '--config', default='default_config.json', help='Configuration of model')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='Batch size of dataloader')
    parser.add_argument('-k', '--top_num', default=100, type=int, help='Num. of predictions to write')
    parser.add_argument('-n', '--n_passes', default=20, type=int, help='Number of MC dropout forward passes per molecule')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    args = parser.parse_args().__dict__
    args['mode'] = 'test'
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device %s' % args['device'])
    main(args)
