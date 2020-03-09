# -*- coding: utf-8 -*-
#
# This script can be used to train any deep learning model on the BigEarthNet. 
#
# To run the code, you need to provide a json file for configurations of the training.
# 
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1
# Usage: train.py [CONFIG_FILE_PATH]

from __future__ import print_function

# TODO - allow randomness during actual training 
SEED = 42

import random as rn
rn.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.set_random_seed(SEED)

from src.data import ZindiDataset
from src.model import VggFcnBaseModel

import os
import argparse
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_model(args):
    with tf.Session() as sess:
        iterator = ZindiDataset(
            TFRecord_paths = args['tr_tf_record_files'], 
            batch_size = args['batch_size'], 
            nb_epoch = args['nb_epoch'], 
            shuffle_buffer_size = args['shuffle_buffer_size'],
        ).batch_iterator
        nb_iteration = int(np.ceil(float(args['training_size'] * args['nb_epoch']) / args['batch_size']))
        iterator_ins = iterator.get_next()

        # load VGG base model and restore weights
        model = VggFcnBaseModel()
        model.load_pretrained(args['model_file'], sess)

        # add segmentation head, only initialize tensors in its scope
        model.build_segmentation_head(session=sess)
        model.define_loss()
        model.define_optimizer(args['learning_rate'], sess)

        # set up model saver
        model_saver = tf.train.Saver(max_to_keep = 5, var_list=tf.global_variables())

        #_, metric_means, metric_update_ops = get_metrics(model.multi_hot_label, model.predictions, model.probabilities)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(args['out_dir'], 'logs', 'training'), sess.graph)

        iteration_idx = 0
        progress_bar = tf.contrib.keras.utils.Progbar(target = nb_iteration, stateful_metrics = ['train_loss', 'val_loss']) 
        while True:
            try:
                batch_dict = sess.run(iterator_ins)
            except tf.errors.OutOfRangeError:
                break
            #_, _, batch_loss, batch_summary = sess.run([train_op, metric_update_ops, model.train_loss, summary_op], 
            _, train_loss, val_loss, batch_summary = sess.run([model.train_op, model.train_loss, model.val_loss, summary_op], 
                                                        feed_dict = model.feed_dict(batch_dict, is_training=True))
            iteration_idx += 1
            summary_writer.add_summary(batch_summary, iteration_idx)
            if (iteration_idx % args['save_checkpoint_per_iteration'] == 0) and (iteration_idx >= args['save_checkpoint_after_iteration']):
                model_saver.save(sess, os.path.join(args['out_dir'], 'models', 'iteration'), iteration_idx)
            progress_bar.update(iteration_idx, values=[('train_loss', train_loss), ('val_loss', val_loss)])
        model_saver.save(sess, os.path.join(args['out_dir'], 'models', 'iteration'), iteration_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Training script')
    parser.add_argument('configs', help= 'json config file')
    parser_args = parser.parse_args()

    with open(os.path.realpath(parser_args.configs), 'rb') as f:
        model_args = json.load(f)

    run_model(model_args)
