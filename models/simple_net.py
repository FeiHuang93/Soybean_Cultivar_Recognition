from .base_net import BaseNet
import tensorflow as tf


class SimpleNet(BaseNet):
    def __init__(self, data_provider,
                 weight_decay, nesterov_momentum, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 **kwargs):
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.dataset_name = dataset
        super().__init__(data_provider, should_save_logs, should_save_model, renew_logs, **kwargs)

    @property
    def model_identifier(self):
        return "SimpleNet_" + self.dataset_name + "_kp="

    def _build_graph(self):
        with tf.variable_scope("init"):
            output = self.conv2d(self.images, 24, 3)
        with tf.variable_scope("conv1_512"):
            output = self.batch_norm(output)
            output = self.relu(output)
            output = self.conv2d(output, 24, 3)
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv2_256"):
            output = self.batch_norm(output)
            output = self.relu(output)
            output = self.conv2d(output, 48, 3)
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv3_128"):
            output = self.batch_norm(output)
            output = self.relu(output)
            output = self.conv2d(output, 96, 3)
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv4_64"):
            output = self.batch_norm(output)
            output = self.relu(output)
            output = self.conv2d(output, 192, 3)
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv5_32"):
            output = self.batch_norm(output)
            output = self.relu(output)
            output = self.conv2d(output, 384, 3)
        with tf.variable_scope("bn_32"):
            output = self.conv2d(output, 192, 1)
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv_16"):
            output = self.batch_norm(output)
            output = self.relu(output)
            output = self.conv2d(output, 384, 3)
        with tf.variable_scope("bn_16"):
            output = self.conv2d(output, 192, 1)
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv_8"):
            output = self.batch_norm(output)
            output = self.relu(output)
            output = self.conv2d(output, 384, 3)
        with tf.variable_scope("bn_8"):
            output = self.conv2d(output, 192, 1)
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv_4"):
            output = self.batch_norm(output)
            output = self.relu(output)
            output = self.conv2d(output, 384, 3)
        with tf.variable_scope("bn_4"):
            output = self.conv2d(output, 192, 1)
            output = self.avg_pool(output, 2)
        with tf.variable_scope("softmax"):
            logits = self.fully_connected(output, self.n_classes)
        prediction = tf.nn.softmax(logits)

        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(
            cross_entropy + l2_loss * self.weight_decay)

        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))