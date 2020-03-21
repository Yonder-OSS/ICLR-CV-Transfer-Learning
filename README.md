# BigEarthNet Deep Learning Models With TensorFlow

## Training
The script `train.py` expects a `JSON` configuration file path as a comand line argument. This file uses the following parameters:
* `batch_size`: Batch size used during training
* `nb_epoch`: The number of epochs for the training
* `learning_rate`: The initial learning rate
* `out_dir`: The path where all log files and checkpoints will be saved.
* `exponential_decay`: whether to exponentially decay the learning rate
* `freeze`: whether to freeze ResNet/VGG backbone during finetuning
* `load_frozen`: whether one is loading a checkpoint from a training run with a frozen backbone (effects optimizer state)
* `save_checkpoint_after_iteration`: The iteration after which checkpoint saving should start, i.e., no checkpoints are saved before. Set to zero to always have checkpoints saved.
* `save_checkpoint_per_iteration`: The number of iterations per which a checkpoint is written, i.e., when `iteration_index % save_checkpoint_per_iteration == 0`.
* `tr_tf_record_files`: An array containing `TFRecord` file(s) for training.
`pretrained_model_path`.
* `model_file`: The base name of a pre-trained model snapshot (i.e., checkpoint).
* `shuffle_buffer_size`: The number of elements which will be shuffled at the beginning of each epoch. It is not recommended to have large shuffle buffer if you don't have enough space in memory. 
* `training_size`: The size of the training set.

## Evaluation
The script `eval.py` expects a `JSON` configuration file. The needed parameters are as follows:
* `batch_size`: Batch size used during evaluation
* `out_dir`: The path where all log files and checkpoints will be saved.
* `test_tf_record_files`: An array containing `TFRecord` files for evaluation.
* `model_file`: The base name of a pre-trained model snapshot (i.e., checkpoint).
* `test_size`: The size of the test set. 

This code is bootstrapped from https://gitlab.tu-berlin.de/rsim/bigearthnet-models-tf

Original Authors
-------

**Gencer Sümbül**
http://www.user.tu-berlin.de/gencersumbul/

**Tristan Kreuziger**
https://www.rsim.tu-berlin.de/menue/team/tristan_kreuziger/

Original License
-------

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
