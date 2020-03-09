# -*- coding: utf-8 -*-
#
# This script can be used to evaluate the performance of a deep learning model, pre-trained on the BigEarthNet.
#
# To run the code, you need to provide the json file which was used for training before. 
# 
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1
# Usage: eval.py [CONFIG_FILE_PATH]

from __future__ import print_function
import numpy as np
import tensorflow as tf
import subprocess, time, os
import argparse
from src.data import ZindiDataset
from src.model import VggFcnBaseModel
from utils import get_metrics
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def eval_model(args):
    with tf.Session() as sess:
        iterator = ZindiDataset(
            TFRecord_paths = args['test_tf_record_files'], 
            batch_size = args['batch_size'], 
            nb_epoch = 1, 
            shuffle_buffer_size = 0,
        ).batch_iterator
        nb_iteration = int(np.ceil(float(args['test_size']) / args['batch_size']))
        iterator_ins = iterator.get_next()

        # load VGG base model and restore weights
        model = VggFcnBaseModel()
        model.load_pretrained(args['model_file'], sess)

        # add segmentation head, only initialize tensors in its scope
        model.build_segmentation_head(session=sess)
        model.define_loss()
        #metric_names, metric_means, metric_update_ops = get_metrics(model.predictions, model.probabilities)
        #summary_op = tf.summary.merge_all()
        #summary_writer = tf.summary.FileWriter(os.path.join(args['out_dir'], 'logs', 'test'), sess.graph)

        iteration_idx = 0
        progress_bar = tf.contrib.keras.utils.Progbar(target=nb_iteration)
        #eval_res = {}
        while True:
            try:
                batch_dict = sess.run(iterator_ins)
                iteration_idx += 1
                progress_bar.update(iteration_idx)
            except tf.errors.OutOfRangeError:
                # means = sess.run(metric_means[0])
                # for idx, name in enumerate(metric_names[0]):
                #     eval_res[name] = str(means[idx])
                #     print(name, means[idx])
                break
            sess_res = sess.run([
                tf.shape(model.logits), 
                tf.reduce_sum(model.train_loss), 
                tf.reduce_sum(model.val_loss)], 
                feed_dict=model.feed_dict(batch_dict, is_training=False)
            )
            print(sess_res)
            # summary_writer.add_summary(sess_res[1], iteration_idx)
        #     metric_means_res = sess_res[2:]

        # for idx, name in enumerate(metric_names[1]):
        #     eval_res[name] = str(metric_means_res[idx])
        #     print(name, metric_means_res[idx])

        # with open(os.path.join(args['out_dir'], 'eval_result.json'), 'wb') as f:
        #     json.dump(eval_res, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test arguments')
    parser.add_argument('settings', help='json settings file')
    parser_args = parser.parse_args()

    with open(os.path.realpath(parser_args.settings), 'rb') as f:
        model_args = json.load(f)

    eval_model(model_args)
