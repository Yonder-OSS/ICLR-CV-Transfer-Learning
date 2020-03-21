from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from .resnet_v1 import resnet_v1_152
from .resnet_utils import resnet_arg_scope

slim = contrib_slim

from .model_utils import weight_variable, bias_variable, conv2d_transpose_strided, conv2d_basic

BAND_STATS = {
            'mean': {
                'B01': 340.76769064,
                'B02': 429.9430203,
                'B03': 614.21682446,
                'B04': 590.23569706,
                'B05': 950.68368468,
                'B06': 1792.46290469,
                'B07': 2075.46795189,
                'B08': 2218.94553375,
                'B8A': 2266.46036911,
                'B09': 2246.0605464,
                'B11': 1594.42694882,
                'B12': 1009.32729131
            },
            'std': {
                'B01': 554.81258967,
                'B02': 572.41639287,
                'B03': 582.87945694,
                'B04': 675.88746967,
                'B05': 729.89827633,
                'B06': 1096.01480586,
                'B07': 1273.45393088,
                'B08': 1365.45589904,
                'B8A': 1356.13789355,
                'B09': 1302.3292881,
                'B11': 1079.19066363,
                'B12': 818.86747235
            }
        }

CLASS_COUNTS = [1462, 829, 98, 487, 172, 160, 78]

class ZindiBaseModel(object):
    """ Base model to define initial placeholders, feed dictionary, and image normalization. 
        Original author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
    """

    def __init__(
        self, 
        nb_class = 8,
    ):       
        self.is_training = tf.placeholder(tf.bool, [])
        self.nb_class = nb_class # dense classification (every pixel) into 8 crop types

        # input Sentinel 2 bands
        self.B01 = tf.placeholder(tf.float32, [None, 120, 120], name='B01')
        self.B02 = tf.placeholder(tf.float32, [None, 120, 120], name='B02')
        self.B03 = tf.placeholder(tf.float32, [None, 120, 120], name='B03')
        self.B04 = tf.placeholder(tf.float32, [None, 120, 120], name='B04')
        self.B05 = tf.placeholder(tf.float32, [None, 120, 120], name='B05')
        self.B06 = tf.placeholder(tf.float32, [None, 120, 120], name='B06')
        self.B07 = tf.placeholder(tf.float32, [None, 120, 120], name='B07')
        self.B08 = tf.placeholder(tf.float32, [None, 120, 120], name='B08')
        self.B8A = tf.placeholder(tf.float32, [None, 120, 120], name='B8A')
        self.B09 = tf.placeholder(tf.float32, [None, 120, 120], name='B09')
        self.B11 = tf.placeholder(tf.float32, [None, 120, 120], name='B11')
        self.B12 = tf.placeholder(tf.float32, [None, 120, 120], name='B12')

        # stack bands
        self.img = tf.stack(
            [
                self.B04, 
                self.B03, 
                self.B02, 
                self.B08,
                self.B05, 
                self.B06, 
                self.B07, 
                self.B8A, 
                self.B11, 
                self.B12
            ],
            axis=3
        )

        # labels
        self.crop_id = tf.placeholder(tf.int32, [None, 120, 120], name='crop_id')
        self.field_id = tf.placeholder(tf.int32, [None, 120, 120, 1], name='field_id')

        # test / val mask (328 validation field ids, 1 test field id)
        self.field_mask = tf.placeholder(tf.int32, [None, 1, 1, 328], name='field_mask')

        # initialize class weights lookup table
        self.class_weights = self._init_class_weights()

    def _init_class_weights(
        self
    ):
        inv_counts = 1 / np.array(CLASS_COUNTS)
        # add 0 weight for crop id 0 (test set)
        return tf.constant(
            np.concatenate((np.array([0]), inv_counts / np.sum(inv_counts))),
            dtype = tf.float32
        )

    def feed_dict(
        self, 
        batch_dict, 
        is_training=False, 
        model_path=''
    ):

        B01  = ((batch_dict['B01'] - BAND_STATS['mean']['B01']) / BAND_STATS['std']['B01']).astype(np.float32)
        B02  = ((batch_dict['B02'] - BAND_STATS['mean']['B02']) / BAND_STATS['std']['B02']).astype(np.float32)
        B03  = ((batch_dict['B03'] - BAND_STATS['mean']['B03']) / BAND_STATS['std']['B03']).astype(np.float32)
        B04  = ((batch_dict['B04'] - BAND_STATS['mean']['B04']) / BAND_STATS['std']['B04']).astype(np.float32)
        B05  = ((batch_dict['B05'] - BAND_STATS['mean']['B05']) / BAND_STATS['std']['B05']).astype(np.float32)
        B06  = ((batch_dict['B06'] - BAND_STATS['mean']['B06']) / BAND_STATS['std']['B06']).astype(np.float32)
        B07  = ((batch_dict['B07'] - BAND_STATS['mean']['B07']) / BAND_STATS['std']['B07']).astype(np.float32)
        B08  = ((batch_dict['B08'] - BAND_STATS['mean']['B08']) / BAND_STATS['std']['B08']).astype(np.float32)
        B8A  = ((batch_dict['B8A'] - BAND_STATS['mean']['B8A']) / BAND_STATS['std']['B8A']).astype(np.float32)
        B09  = ((batch_dict['B09'] - BAND_STATS['mean']['B09']) / BAND_STATS['std']['B09']).astype(np.float32)
        B11  = ((batch_dict['B11'] - BAND_STATS['mean']['B11']) / BAND_STATS['std']['B11']).astype(np.float32)
        B12  = ((batch_dict['B12'] - BAND_STATS['mean']['B12']) / BAND_STATS['std']['B12']).astype(np.float32)

        return {
                self.B01: B01,
                self.B02: B02,
                self.B03: B03,
                self.B04: B04,
                self.B05: B05,
                self.B06: B06,
                self.B07: B07,
                self.B08: B08,
                self.B8A: B8A,
                self.B09: B09,
                self.B11: B11,
                self.B12: B12,
                self.is_training:is_training,
                self.crop_id:batch_dict['crop_id'],
                self.field_id:batch_dict['field_id'],
                self.field_mask:batch_dict['field_mask'],
            }

    def define_loss(
        self,
        l2_regularization = 0.001,
        freeze = True,
    ):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.crop_id, logits=self.logits)
        
        # mask loss w/ test set and val set ids
        test_mask = tf.math.not_equal(self.crop_id, 0)
        test_mask = tf.cast(test_mask, tf.float32)
        masked_loss = tf.multiply(loss, test_mask, name='masked_loss')

        # weight loss according to class weights
        # class_weight_mask = tf.gather(self.class_weights, self.crop_id)
        # weighted_loss = tf.multiply(masked_loss, class_weight_mask, name='weighted_loss')

        train_loss_mask = tf.reduce_all(tf.math.not_equal(self.field_id, self.field_mask), -1)
        train_loss_mask = tf.cast(train_loss_mask, tf.float32)
        train_loss = tf.multiply(masked_loss, train_loss_mask, name='train_loss')
        self.train_loss = tf.divide(
            tf.reduce_sum(train_loss), 
            tf.count_nonzero(train_loss, dtype=tf.float32)
        )

        # val loss
        val_loss_mask = tf.reduce_any(tf.math.equal(self.field_id, self.field_mask), -1)
        val_loss_mask = tf.cast(val_loss_mask, tf.float32)
        val_loss = tf.multiply(masked_loss, val_loss_mask, name='val_loss')
        self.val_loss = tf.divide(
            tf.reduce_sum(val_loss), 
            tf.count_nonzero(val_loss, dtype=tf.float32)
        )

        # add l2_loss
        if l2_regularization:
            if freeze:
                self.l2_loss = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fcn_seg") if 'bias' not in v.name]
                ) * l2_regularization
            else:
                self.l2_loss = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]
                ) * l2_regularization
        self.train_loss += self.l2_loss
        self.val_loss += self.l2_loss

        # write metrics to tb
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('train_loss', self.train_loss)
        tf.summary.scalar('val_loss', self.val_loss)

    def define_optimizer(
        self, 
        session,
        starter_learning_rate, 
        freeze = True,
        exponential_decay = True,
        decay_steps = 5,
        decay_rate = 0.96,
        staircase = True
    ):
        
        # define exponential learning rate schedule
        if exponential_decay:
            global_step = tf.Variable(0, trainable = False)
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate,
                global_step,
                decay_steps,
                decay_rate,
                staircase = staircase
            )
        else:
            learning_rate = starter_learning_rate

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            if freeze:
                if exponential_decay:
                    self.train_op = optimizer.minimize(
                        self.train_loss, 
                        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fcn_seg"),
                        global_step = global_step
                    )
                else:
                    self.train_op = optimizer.minimize(
                        self.train_loss, 
                        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fcn_seg")
                    ) 
            else:
                if exponential_decay:
                    self.train_op = optimizer.minimize(
                        self.train_loss, 
                        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vgg_16"),
                        global_step = global_step
                    )
                else:
                    self.train_op = optimizer.minimize(
                        self.train_loss, 
                        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vgg_16")
                    )
                # if exponential_decay:
                #     self.train_op = optimizer.minimize(self.train_loss, global_step = global_step)
                # else:
                #     self.train_op = optimizer.minimize(self.train_loss)

        # initialize variables
        if exponential_decay:
            session.run(tf.variables_initializer(optimizer.variables() + [global_step]))
        else:
            session.run(tf.variables_initializer(optimizer.variables()))

def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      #biases_initializer=tf.compat.v1.zeros_initializer()):
                      weights_initializer=slim.xavier_initializer(),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

class VggFcnBaseModel(ZindiBaseModel):

    def __init__(
        self,
        nb_class = 8,
    ):
        super(VggFcnBaseModel, self).__init__(
            nb_class=nb_class,
        )

        with slim.arg_scope(vgg_arg_scope()):
            fc7, end_points = self._vgg_16(
                self.img,
                num_classes = 0,
                is_training = self.is_training,
                fc_conv_padding = 'SAME',
                global_pool=False
            )
        self.fc7 = fc7

    def _vgg_16(
        self,
        inputs,
        num_classes=1000,
        is_training=True,
        dropout_keep_prob=0.5,
        spatial_squeeze=False,
        reuse=None,
        scope='vgg_16',
        fc_conv_padding='VALID',
        global_pool=False
    ):

        """Oxford Net VGG 16-Layers version D Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
                To use in classification mode, resize input to 224x224.

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            num_classes: number of predicted classes. If 0 or None, the logits layer is
            omitted and the input features to the logits layer are returned instead.
            is_training: whether or not the model is being trained.
            dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
            spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
            reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
            scope: Optional scope for the variables.
            fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.
            global_pool: Optional boolean flag. If True, the input to the classification
            layer is avgpooled to size 1x1, for any input size. (This is not part
            of the original VGG architecture.)

        Returns:
            net: the output of the logits layer (if num_classes is a non-zero integer),
            or the input to the logits layer (if num_classes is 0 or None).
            end_points: a dict of tensors with intermediate activations.
        """
        #with tf.compat.v1.variable_scope(
        with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                self.pool3 = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(self.pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                self.pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(self.pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                    scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(
                        input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                    scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='fc8')

                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points

    def load_pretrained(
        self,
        model_file,
        session,
    ):
        """ loads pretrained model from model file 
        """
    
        variables_to_restore = tf.global_variables()
        # session.run(tf.global_variables_initializer())
        # session.run(tf.local_variables_initializer())
        model_saver = tf.train.Saver(var_list=variables_to_restore)
        model_saver.restore(session, model_file)

    def build_segmentation_head(
        self,
        session,
        dropout_keep_prob=0.5,
        #weight_decay=0.0005,
        scope='fcn_seg',
        reuse=None
    ):
        """ builds decoder that upscales to actual image
        
            Args:
                reuse: whether or not the network and its variables should be reused. To be
                able to reuse 'scope' must be given.
                scope: Optional scope for the variables.
        """
        
        with tf.variable_scope(scope, 'fcn_seg', [self.fc7], reuse=reuse) as sc:

            # last fully convolutional layer that maps 4096 dimensions -> # classes (might be different then 
            # number of classes on which network was trained for image segmentation)
            dropout7 = slim.dropout(self.fc7, dropout_keep_prob, is_training=self.is_training, scope='dropout7') 
            W8 = weight_variable([1, 1, 4096, self.nb_class], name="weight_8")
            b8 = bias_variable([self.nb_class], name="bias_8")
            fc8 = conv2d_basic(dropout7, W8, b8)

            # upsample last convolutional output to coarse-ness of pool4 vgg layer, sum these two tensors
            deconv_shape1 = self.pool4.get_shape()
            W_t1 = weight_variable([2, 2, deconv_shape1[3].value, self.nb_class], name="weight_t1")
            b_t1 = bias_variable([deconv_shape1[3].value], name="bias_t1")
            conv_t1 = conv2d_transpose_strided(fc8, W_t1, b_t1, output_shape=tf.shape(self.pool4))
            fuse_1 = tf.add(conv_t1, self.pool4, name="fuse_1")

            # upsample previously fused tensor to coarse-ness of pool3 vgg layer, sum these two tensors
            deconv_shape2 = self.pool3.get_shape()
            W_t2 = weight_variable([3, 3, deconv_shape2[3].value, deconv_shape1[3].value], name="weight_t2")
            b_t2 = bias_variable([deconv_shape2[3].value], name="bias_t2")
            conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(self.pool3))
            fuse_2 = tf.add(conv_t2, self.pool3, name="fuse_2")

            # upsample previously fused tensor to coarse-ness of input image
            shape = tf.shape(self.img)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.nb_class])
            W_t3 = weight_variable([8, 8, self.nb_class, deconv_shape2[3].value], name="weight_t3")
            b_t3 = bias_variable([self.nb_class], name="bias_t3")
            conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

            self.logits = conv_t3

        # intialize new variables added to the graph 
        session.run(tf.variables_initializer([W8, b8, W_t1, b_t1, W_t2, b_t2, W_t3, b_t3]))


class Resnet152FcnBaseModel(ZindiBaseModel):

    def __init__(
        self,
        nb_class = 8,
        freeze = False,
    ):
        super(Resnet152FcnBaseModel, self).__init__(
            nb_class=nb_class,
            freeze=freeze
        )

        with slim.arg_scope(resnet_arg_scope()):
            fc7, end_points = resnet_v1_152(
                self.img,
                num_classes = 0,
                is_training = self.is_training,
                global_pool=False,
                spatial_squeeze = True, #TODO can this be false
            )
        self.fc7 = fc7

    def load_pretrained(
        self,
        model_file,
        session,
    ):
        """ loads pretrained model from model file 
        """
    
        variables_to_restore = tf.global_variables()
        # session.run(tf.global_variables_initializer())
        # session.run(tf.local_variables_initializer())
        model_saver = tf.train.Saver(var_list=variables_to_restore)
        model_saver.restore(session, model_file)

    def build_segmentation_head(
        self,
        session,
        dropout_keep_prob=0.5,
        #weight_decay=0.0005,
        scope='fcn_seg',
        reuse=None
    ):
        """ builds decoder that upscales to actual image
        
            Args:
                reuse: whether or not th e network and its variables should be reused. To be
                able to reuse 'scope' must be given.
                scope: Optional scope for the variables.
        """
        
        with tf.variable_scope(scope, 'fcn_seg', [self.fc7], reuse=reuse) as sc:

            # last fully convolutional layer that maps 4096 dimensions -> # classes (might be different then 
            # number of classes on which network was trained for image segmentation)
            dropout7 = slim.dropout(self.fc7, dropout_keep_prob, is_training=self.is_training, scope='dropout7') 
            W8 = weight_variable([1, 1, 2048, self.nb_class], name="weight_8")
            b8 = bias_variable([self.nb_class], name="bias_8")
            fc8 = conv2d_basic(dropout7, W8, b8)            

            # upsample last convolutional output to coarse-ness of pool4 vgg layer, sum these two tensors
            # deconv_shape1 = self.pool4.get_shape()
            # W_t1 = weight_variable([2, 2, deconv_shape1[3].value, self.nb_class], name="weight_t1")
            # b_t1 = bias_variable([deconv_shape1[3].value], name="bias_t1")
            # conv_t1 = conv2d_transpose_strided(fc8, W_t1, b_t1, output_shape=tf.shape(self.pool4))
            # fuse_1 = tf.add(conv_t1, self.pool4, name="fuse_1")

            # # upsample previously fused tensor to coarse-ness of pool3 vgg layer, sum these two tensors
            # deconv_shape2 = self.pool3.get_shape()
            # W_t2 = weight_variable([2, 2, deconv_shape2[3].value, deconv_shape1[3].value], name="weight_t2")
            # b_t2 = bias_variable([deconv_shape2[3].value], name="bias_t2")
            # conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(self.pool3))
            # fuse_2 = tf.add(conv_t2, self.pool3, name="fuse_2")

            # upsample previously fused tensor to coarse-ness of input image
            shape = tf.shape(self.img)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.nb_class])
            W_t3 = weight_variable([2, 2, self.nb_class, fc8.get_shape()[3].value], name="weight_t3")
            b_t3 = bias_variable([self.nb_class], name="bias_t3")
            # output = tf.constant(0.1, shape=(10,120,120,8))
            # expected_l = tf.nn.conv2d(output, W_t3, strides=[1,32,32,1], padding='VALID')
            # print(expected_l.get_shape())
            conv_t3 = conv2d_transpose_strided(fc8, W_t3, b_t3, output_shape=deconv_shape3, stride=32)
            self.logits = conv_t3

        # intialize new variables added to the graph 
        session.run(tf.variables_initializer([W8, b8, W_t3, b_t3]))
    
