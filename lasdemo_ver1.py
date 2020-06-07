from mosselas5_CACF import mosselas5_CACF
import argparse
import os
import numpy as np
import tensorflow as tf 
from tensorflow import keras

parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.125, help='the learning rate')
parse.add_argument('--sigma', type=float, default=100, help='the sigma')
parse.add_argument('--num_pretrain', type=int, default=128, help='the number of pretrain')
parse.add_argument('--rotate', action='store_true', help='if rotate image during pre-training.')
parse.add_argument('--record', action='store_true', help='record the frames')

if __name__ == '__main__':
    args = parse.parse_args()
    all_dataset = os.listdir("F:/git_folder/correlation_with_resnet50/datasets")
    total_number_of_datasets = len(all_dataset) 
    iou_of_each_dataset = []
    for available_datasets in range(0,total_number_of_datasets):
        dataset = all_dataset[available_datasets]
        dataset1 = 'Biker'
        img_path = 'datasets/'+ dataset1  
        tracker = mosselas5_CACF(args, img_path, dataset1)
        val = tracker.start_tracking()
        #overall_iou = tracker.iou_computation(val)
        overall_iou = tracker.iou_computation_CACFmethod(val)
        print(overall_iou)
        iou_of_each_dataset.append(overall_iou)
    print (np.mean(iou_of_each_dataset)) 
