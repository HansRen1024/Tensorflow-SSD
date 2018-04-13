#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

IMAGE_RESIZE_SHAPE = 300 # image shape which suits for network
NUM_LABELS = 21 # number of classes
MEAN = [123.68, 116.78, 103.94] # BGR mean values of training dataset

parser = argparse.ArgumentParser()

INITIAL_LEARNING_RATE = 0.01
DEBUG = False
DATASET_DIR = 'voc2012/' # Path to data directory.
MODEL_DIR = 'models/' # Directory where to write event logs and checkpoint.
FINETUNE_DIR = None
BATCH_SIZE = 4
LOG_FREQUENCY = 10 # How often to log results to the console.
MAX_STEPS = 10000 # Number of batches to run. If distributiong, all GPU batches.
MODE = 'training'
ISSYNC = False # only for distribution

parser.add_argument('--mode', type=str,default=MODE, help='Either `training` or `testing`.')
parser.add_argument('--lr', type=float, default=INITIAL_LEARNING_RATE)
parser.add_argument('--debug', type=bool, default=DEBUG)
parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
parser.add_argument('--finetune', type=str, default=FINETUNE_DIR)
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('--log_frequency', type=int, default=LOG_FREQUENCY)
parser.add_argument('--max_steps', type=int, default=MAX_STEPS)

# For distributed
PS_HOSTS = '10.100.3.101:2222' # Comma-separated list of hostname:port pairs
WORKER_HOSTS = '10.100.3.101:2224,10.100.3.100:2225,10.107.3.10:2226' # Comma-separated list of hostname:port pairs

parser.add_argument('--issync', type=bool, default=ISSYNC)
parser.add_argument("--job_name", type=str,
                    help="One of 'ps', 'worker'")
parser.add_argument("--task_index", type=int,
                    help="Index of task within the job")
