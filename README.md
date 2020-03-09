# BigEarthNet Deep Learning Models With TensorFlow

## Training
The script `train.py` expects a `JSON` configuration file path as a comand line argument. This file contains the following parameters:
* `model_name`: The name of the Python code containing the corresponding deep learning model. The code must be located under the `models` directory. The model class will be loaded dynamically based on the `model_name` parameter: `model = importlib.import_module('models.' + args['model_name']).dnn_model(nb_class)`
* `label_type`: A flag to indicate which labels will be used during training: `original` or `BigEarthNet-19`
* `batch_size`: Batch size used during training
* `nb_epoch`: The number of epochs for the training
* `learning_rate`: The initial learning rate
* `out_dir`: The path where all log files and checkpoints will be saved.
* `save_checkpoint_after_iteration`: The iteration after which checkpoint saving should start, i.e., no checkpoints are saved before. Set to zero to always have checkpoints saved.
* `save_checkpoint_per_iteration`: The number of iterations per which a checkpoint is written, i.e., when `iteration_index % save_checkpoint_per_iteration == 0`.
* `tr_tf_record_files`: An array containing `TFRecord` file(s) for training.
* `val_tf_record_files`: An array containing `TFRecord` file(s) for validation (not used for now).
* `fine_tune`: A flag to indicate if the training of the model will continue from the existing checkpoint whose path will be defined by `pretrained_model_path`.
* `model_file`: The base name of a pre-trained model snapshot (i.e., checkpoint).
* `shuffle_buffer_size`: The number of elements which will be shuffled at the beginning of each epoch. It is not recommended to have large shuffle buffer if you don't have enough space in memory. 
* `training_size`: The size of the training set. If you are using training set suggested in [here](https://gitlab.tu-berlin.de/rsim/bigearthnet-models/), it is already set.

## Evaluation
The script `eval.py` expects a `JSON` configuration file. The needed parameters are as follows:
* `model_name`: The name of the Python code containing the corresponding deep learning model. The code must be located under the `models` directory. The model class will be loaded dynamically based on the `model_name` parameter: `model = importlib.import_module('models.' + args['model_name']).dnn_model(nb_class)`
* `batch_size`: Batch size used during evaluation
* `out_dir`: The path where all log files and checkpoints will be saved.
* `test_tf_record_files`: An array containing `TFRecord` files for evaluation.
* `model_file`: The base name of a pre-trained model snapshot (i.e., checkpoint).
* `test_size`: The size of the test set. If you are using test set suggested in [here](https://gitlab.tu-berlin.de/rsim/bigearthnet-models/), it is already set.

Original Authors
-------

This code is bootstrapped from https://gitlab.tu-berlin.de/rsim/bigearthnet-models-tf

**Gencer Sümbül**
http://www.user.tu-berlin.de/gencersumbul/

**Tristan Kreuziger**
https://www.rsim.tu-berlin.de/menue/team/tristan_kreuziger/

# License
The code in this repository to facilitate the use of the BigEarthNet archive is licensed under the **MIT License**:

```
MIT License

Copyright (c) 2019 The BigEarthNet Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
