# -*- coding:utf-8 -*-

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='RKLF-Graph2Tree')
    parser.add_argument('--cuda', type=str, dest='cuda_id', default=None)
    args = parser.parse_args()
    return args

args = get_args()