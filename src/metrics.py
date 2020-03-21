import time

from sklearn.metrics import log_loss
from scipy.special import softmax
import numpy as np
import pandas as pd
import tensorflow as tf

# def softmax_2(logits):
#     e_x = np.exp(logits - np.max(logits))
#     s = e_x.sum(axis=1).reshape(-1, 1)
#     return e_x / s

class InferenceAggregation(object):
    """
        aggregates logits by field to calculate metrics for ICLR CV challenge
    """

    def __init__(
        self,
        all_fields_path = '/root/data/all_fields.npy',
        val_set_path = '/root/data/val_set.npy',
        num_classes = 7
    ):

        ## load field val_set list from .npy file
        self.val_set = np.load(val_set_path)

        # ignore field id 0
        all_fields = np.load(all_fields_path)

        ## create dictionaries for summed logits, ground truth
        self.train_test_logits = {k: [-1, None] for k in all_fields if k != 0}
        self.val_logits = {k: [-1, None] for k in self.val_set}

        self.pixel_logits = dict()
        # self.pixel_gt = dict()
        self.counter = 0

    def _append(self, d, logit, f_id, c_id):

        # only initial zeros array exists
        if d[f_id][1] is None:
            d[f_id][1] = logit
        else:
            d[f_id][1] = np.concatenate((d[f_id][1], logit), axis = 0)

        d[f_id][0] = c_id

        return d

    def process(
        self,
        logits,
        crop_id,
        field_id,
    ):

        # ignore log odds probability mass from crop id 0
        logits = np.reshape(logits[:, :, :, 1:], (logits.shape[0]*120*120, 7))
        crop_id = crop_id.flatten()
        field_id = field_id.flatten()

        for logit, c_id, f_id in zip(logits, crop_id, field_id):

            # 1 row, n_classes cols
            logit = logit.reshape(1, -1)

            # ignore field id 0 (nothing)
            if f_id != 0:
                if f_id in self.val_set:
                    self.val_logits = self._append(self.val_logits, logit, f_id, c_id)
                else:
                    self.train_test_logits = self._append(self.train_test_logits, logit, f_id, c_id)

                # pixel logits
                if c_id != 0:
                    self.pixel_logits[self.counter] = [c_id, logit]
                    self.counter += 1
        
    def _agg(
        self, 
        vals
    ):

        # filter out f_ids that were never updated - all should be updated, this is probably unnecessary
        vals = [v for _, v in vals.items() if v[1] is not None]

        # different aggregations
        gt, preds = map(list, zip(*vals))
        sum_preds = [np.sum(p, axis=0) for p in preds]
        mean_preds = [np.mean(p, axis=0) for p in preds]
        median_preds = [np.median(p, axis=0) for p in preds]
        
        sum_nll = log_loss(gt, softmax(sum_preds, axis=1), labels=np.arange(1,8))
        mean_nll = log_loss(gt, softmax(mean_preds, axis=1), labels=np.arange(1,8))
        median_nll = log_loss(gt, softmax(median_preds, axis=1), labels=np.arange(1,8))

        return sum_nll, mean_nll, median_nll

    def aggregate(
        self, 
        experiment_name
    ):    

        # train NLL
        train_logits = {k: v for k, v in self.train_test_logits.items() if v[0] > 0}
        sum_nll, mean_nll, median_nll = self._agg(train_logits)
        print('Mean NLL on training fields: {}'.format(mean_nll))
        print('Median NLL on training fields: {}'.format(median_nll))
        f = open('metrics.txt', 'a')
        f.write('Experiment: {}\n'.format(experiment_name))
        f.write('Mean NLL on training fields: {}\n'.format(mean_nll))
        f.write('Median NLL on training fields: {}\n'.format(median_nll))

        # val NLL
        sum_nll, mean_nll, median_nll = self._agg(self.val_logits)
        print('Mean NLL on validation fields: {}'.format(mean_nll))
        print('Median NLL on validation fields: {}'.format(median_nll))
        f.write('Mean NLL on validation fields: {}\n'.format(mean_nll))
        f.write('Median NLL on validation fields: {}\n'.format(median_nll))

        # pixel NLL
        pixel_gt, pixel_preds = map(list, zip(*self.pixel_logits.values()))
        mean_nll_pixel = log_loss(pixel_gt, softmax(np.squeeze(pixel_preds), axis=1), labels=np.arange(1,8))
        print('Mean NLL on individual pixels: {}'.format(mean_nll_pixel))
        f.write('Mean NLL on individual pixels: {}\n'.format(mean_nll_pixel))
        f.close()

    def write(
        self,
        submission_outfile,
        sample_submission_file = '/root/data/SampleSubmission.csv'
    ):

        # load sample submission csv
        sample = pd.read_csv(sample_submission_file)
        prob_df = pd.DataFrame.from_dict(self.softmax_logits, orient='index')
        prob_df.index.name = 'Field_ID'
        submission_df = pd.concat(sample['Field_ID'], prob_df, how='left', on='Field_ID')

        print('There are {} missing fields'.format(submission_df.isna().sum())) # Missing fields
        submission_df.to_csv(submission_outfile, index=False)

