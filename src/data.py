import numpy as np
import glob
from tqdm.auto import tqdm
import tensorflow as tf

# Step 1: Split time dimension into separate samples - disregarding time dimension for now
    # data = 3237 unique images (249 tiles x 13 timestamps)
    # each sample: [120, 120, 10] or [120, 120, 12]

# Step 2: Reserve 10% of training set for validation (set state for reproducibility)
    # have crop_id and field_id raster images for each image
    # crop_id == 0 is test set
    # randomly sample 10% of field_ids as val set (from subset of fields where crop_id > 0)

# List of Sentinel-2 bands in the dataset
BANDS_IDXS = {
    'B01': 0, 
    'B02': 1, 
    'B03': 2, 
    'B04': 3, 
    'B05': 4, 
    'B06': 5, 
    'B07': 6, 
    'B08': 7, 
    'B8A': 8, 
    'B09': 9, 
    'B11': 10, 
    'B12': 11
}

def scale(array, scale = 10**4):
    """ working assumption is Zindi challenge observations are 10^-4 smaller in magnitude than BigEarth obs"""
    scaled_array = array * scale
    return scaled_array.astype(int)

def expand(array, length = 20):
    """ Big Earth takes input tensors that are either 20x20, 60x60, or 120x120 grids of pixels, so we repeat 
        values to this dimension
    """
    return np.repeat(array, length * length)

def image_to_tf_record(image_tensor, crop_id, field_id):
    """ write image tensor, crop_id array, and field id array (at one timestep, representing one example) 
        to TFRecords format"""

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'B01': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B01']])))),
                'B02': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B02']])))),
                'B03': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B03']])))),
                'B04': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B04']])))),
                'B05': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B05']])))),
                'B06': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B06']])))),
                'B07': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B07']])))),
                'B08': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B08']])))),
                'B8A': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B8A']])))),
                'B09': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B09']])))),
                'B11': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B11']])))),
                'B12': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(scale(image_tensor[BANDS_IDXS['B12']])))),
                'crop_id': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(crop_id))),
                'field_id': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(field_id))),
            }
        )
    )

def prep_data(
    raw_data_path = '/root/data/eopatches-with-reference',
    save_data_path = '/root/data/tfrecords/data.tfrecord'
):
    """ Preps raw data and saves it to TFRecords """

    crop_ids, field_ids = [], []
    eopatch_names = glob.glob('{}/*'.format(raw_data_path))
    pbar = tqdm(total=len(eopatch_names))
    # writer = tf.python_io.TFRecordWriter(save_data_path)

    for eopatch in eopatch_names:

        # load data without eo patch library
        # band_data = np.load('{}/data/S2-BANDS-L2A.npy'.format(eopatch))
        # crop_id_data = np.load('{}/mask_timeless/CROP_ID.npy'.format(eopatch)).squeeze()
        field_id_data = np.load('{}/mask_timeless/FIELD_ID.npy'.format(eopatch)).squeeze()

        # image_timeseries = np.moveaxis(band_data, -1, 1)

        # # expand over time dimension Shape: [Time, Bands, Height, Width]
        # for image_tensor in image_timeseries:
        #     example = image_to_tf_record(
        #         image_tensor, 
        #         crop_id_data, 
        #         field_id_data
        #     )
        #     writer.write(example.SerializeToString())

        field_ids.append(field_id_data)
        #crop_ids.append(crop_id_data)
        
        pbar.update()
        
    field_ids = np.concatenate(field_ids)
    #crop_ids = np.concatenate(crop_ids)
    return field_ids#, crop_ids

def validation_split(
    field_ids, 
    crop_ids,
    val_frac = 0.1,
    random_state = 7
):
    """ Reserve a percentage of field ids for validation (set state for reproducibility) """

    fid = field_ids[crop_ids > 0]
    uniq_field_ids = np.unique(fid)
    np.random.seed(random_state)
    return np.random.choice(uniq_field_ids, int(val_frac * len(uniq_field_ids)), replace=False)

class ZindiDataset:
    """ class to create tf.data.Dataset based on the TFRecord files. """

    def __init__(
        self, 
        TFRecord_paths = '/root/data/tfrecords/data.tfrecord', 
        val_mask_path = '/root/data/val_set.npy', 
        batch_size = 1000, 
        nb_epoch = 100, 
        shuffle_buffer_size = 5000
    ):
        
        # load validation set mask
        field_mask = tf.constant(np.load(val_mask_path))
        self.field_mask = tf.expand_dims(tf.expand_dims(field_mask, 0), 0)

        dataset = tf.data.TFRecordDataset(TFRecord_paths)
        if shuffle_buffer_size > 0:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(nb_epoch)

        dataset = dataset.map(
            lambda x: self.parse_function(x), 
            num_parallel_calls=10
        ) 

        dataset = dataset.batch(batch_size, drop_remainder=False)
        self.dataset = dataset.prefetch(10)
        self.batch_iterator = self.dataset.make_one_shot_iterator()

    def parse_function(self, example_proto):
        parsed_features = tf.parse_single_example(
                example_proto, 
                {
                    'B01': tf.FixedLenFeature([120*120], tf.int64),
                    'B02': tf.FixedLenFeature([120*120], tf.int64),
                    'B03': tf.FixedLenFeature([120*120], tf.int64),
                    'B04': tf.FixedLenFeature([120*120], tf.int64),
                    'B05': tf.FixedLenFeature([120*120], tf.int64),
                    'B06': tf.FixedLenFeature([120*120], tf.int64),
                    'B07': tf.FixedLenFeature([120*120], tf.int64),
                    'B08': tf.FixedLenFeature([120*120], tf.int64),
                    'B8A': tf.FixedLenFeature([120*120], tf.int64),
                    'B09': tf.FixedLenFeature([120*120], tf.int64),
                    'B11': tf.FixedLenFeature([120*120], tf.int64),
                    'B12': tf.FixedLenFeature([120*120], tf.int64),
                    'crop_id': tf.FixedLenFeature([120*120], tf.int64),
                    'field_id': tf.FixedLenFeature([120*120], tf.int64),
                }
            )

        return {
            'B01': tf.reshape(parsed_features['B01'], [120, 120]),
            'B02': tf.reshape(parsed_features['B02'], [120, 120]),
            'B03': tf.reshape(parsed_features['B03'], [120, 120]),
            'B04': tf.reshape(parsed_features['B04'], [120, 120]),
            'B05': tf.reshape(parsed_features['B05'], [120, 120]),
            'B06': tf.reshape(parsed_features['B06'], [120, 120]),
            'B07': tf.reshape(parsed_features['B07'], [120, 120]),
            'B08': tf.reshape(parsed_features['B08'], [120, 120]),
            'B8A': tf.reshape(parsed_features['B8A'], [120, 120]),
            'B09': tf.reshape(parsed_features['B09'], [120, 120]),
            'B11': tf.reshape(parsed_features['B11'], [120, 120]),
            'B12': tf.reshape(parsed_features['B12'], [120, 120]),
            'crop_id': tf.reshape(parsed_features['crop_id'], [120, 120]),
            'field_id': tf.expand_dims(tf.reshape(parsed_features['field_id'], [120, 120]), -1),
            'field_mask': self.field_mask,
        }