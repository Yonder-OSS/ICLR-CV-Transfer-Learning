# This script can be used to evaluate the performance of a deep learning model, pre-trained on the BigEarthNet.
#
# To run the code, you need to provide the json file which was used for training before. 
# 
# Original Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/ gencer.suembuel@tu-berlin.de
# Usage: eval.py [CONFIG_FILE_PATH]

from __future__ import print_function
import numpy as np
import tensorflow as tf
import subprocess, time, os
import argparse
from src.data import ZindiDataset
from src.model import VggFcnBaseModel#, Resnet152FcnBaseModel
from src.metrics import InferenceAggregation
import json
import pickle
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def eval_model(args):
    with tf.Session() as sess:

        # check if GPU is available
        print("GPU is available: {}".format(tf.test.is_gpu_available()))

        iterator = ZindiDataset(
            TFRecord_paths = args['test_tf_record_files'], 
            batch_size = args['batch_size'], 
            nb_epoch = 1, 
            shuffle_buffer_size = 0,
        ).batch_iterator
        nb_iteration = int(np.ceil(float(args['test_size']) / args['batch_size']))
        iterator_ins = iterator.get_next()

        # load VGG base model
        model = VggFcnBaseModel()

        # add segmentation head, and restore weights
        model.build_segmentation_head(session=sess)
        model.define_loss()
        model.load_pretrained(args['model_file'], sess)
    
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(args['out_dir'], 'logs', 'test'), sess.graph)

        # aggregate Inference metrics
        inference_metrics = InferenceAggregation()

        iteration_idx = 0
        progress_bar = tf.contrib.keras.utils.Progbar(target=nb_iteration)

        while True:
            try:
                batch_dict = sess.run(iterator_ins)
                iteration_idx += 1
                progress_bar.update(iteration_idx)
            except tf.errors.OutOfRangeError:
                break

            logits, crop_ids, field_ids, train_loss, val_loss, batch_summary = sess.run(
                [
                    model.logits, 
                    model.crop_id,
                    model.field_id,
                    model.train_loss, 
                    model.val_loss,
                    summary_op
                ], 
                feed_dict=model.feed_dict(batch_dict, is_training=False)
            )
            inference_metrics.process(logits, crop_ids, field_ids)

        # aggregate
        inference_metrics.aggregate(args['model_file'])

        # write log odds for each field in test set to csv
        #inference_metrics.write(os.path.join(args['out_dir'], 'submission.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test arguments')
    parser.add_argument('settings', help='json settings file')
    parser_args = parser.parse_args()

    with open(os.path.realpath(parser_args.settings), 'rb') as f:
        model_args = json.load(f)

    eval_model(model_args)
