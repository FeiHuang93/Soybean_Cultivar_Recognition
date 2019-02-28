from datetime import timedelta
import time
import tensorflow as tf
import numpy as np
import os
import shutil

class BaseNet:
    def __init__(self, data_provider,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 **kwargs):
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes

        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        if tf_ver <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logs_writer = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logs_writer = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logs_writer(self.logs_path)
        self.summary_writer.add_graph(self.sess.graph)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        raise NotImplementedError
        # return "{}_growth_rate={}_depth={}_dataset_{}".format(
        #     self.model_type, self.growth_rate, self.depth, self.dataset_name)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        # reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        # reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        reduce_lr_epoch = train_params['reduce_lr_epoch']
        total_start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch in reduce_lr_epoch:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)

            print("Training...")
            loss, acc = self.train_one_epoch(
                self.data_provider.train, batch_size, learning_rate)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                print("Validation...")
                loss, acc = self.test(
                    self.data_provider.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):

            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_step, self.cross_entropy, self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            # print("batch{}".format(i))
            # print("loss:{}, accuracy:{}".format(loss, accuracy))
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def test(self, data, batch_size):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.is_training: False,
            }
            fetches = [self.cross_entropy, self.accuracy]
            loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME', weight_initializer=None):
        in_features = int(_input.get_shape()[-1])
        if weight_initializer is not None:
            kernel = weight_initializer([kernel_size, kernel_size, in_features, out_features], name="kernel")
        else:
            kernel = self.weight_variable_msra(
                [kernel_size, kernel_size, in_features, out_features],
                name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def relu(self, _input):
        return tf.nn.relu(_input)

    def conv2d_b(self, _input, out_features, kernel_size,
                 strides=[1, 1, 1, 1], padding='SAME', weight_initializer=None):
        in_features = int(_input.get_shape()[-1])
        if weight_initializer is not None:
            kernel = weight_initializer([kernel_size, kernel_size, in_features, out_features], name="kernel")
        else:
            kernel = self.weight_variable_msra(
                [kernel_size, kernel_size, in_features, out_features],
                name='kernel')
        bias = self.bias_variable([out_features], init_value=0.1)
        output = tf.nn.conv2d(_input, kernel, strides, padding) + bias
        return output

    def fully_connected(self, _input, out_features, weight_initializer=None):
        # in_features = int(_input.get_shape()[-1])
        shape = _input.get_shape().as_list()
        in_features = 1
        for d in shape[1:]:
            in_features *= d
        _input = tf.reshape(_input, [-1, in_features])
        if weight_initializer is not None:
            W = weight_initializer([in_features, out_features], name="W")
        else:
            W = self.weight_variable_msra(
                [in_features, out_features],
                name='W')
        bias = self.bias_variable([out_features], init_value=0.1)
        output = tf.matmul(_input, W) + bias
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def max_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.max_pool(_input, ksize, strides, padding)
        return output

    def max_pool_2(self, _input, k, s):
        ksize = [1, k, k, 1]
        strides = [1, s, s, 1]
        padding = 'VALID'
        output = tf.nn.max_pool(_input, ksize, strides, padding)
        return output

    def lrn(self, _input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(_input, radius, bias, alpha, beta, name)

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def weight_variable_tn(self, shape, name, stddev=0.1):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev, name=name))

    def bias_variable(self, shape, init_value=0.0, name='bias'):
        initial = tf.constant(init_value, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
        raise NotImplementedError
