from models.base_net import BaseNet
import tensorflow as tf


class VGG16Net(BaseNet):
    def __init__(self,  data_provider, weight_decay, nesterov_momentum, keep_prob, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False, **kwargs):
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.dataset_name = dataset
        self.keep_prob = keep_prob
        super().__init__(data_provider, should_save_logs, should_save_model, renew_logs, **kwargs)

    @property
    def model_identifier(self):
        return "VGG-16_" + self.dataset_name + "_kp=" + str(self.keep_prob)

    def _build_graph(self):
        with tf.variable_scope("conv1_224x224"):
            output = self.conv2d_b(self.images, 64, 3)
            output = self.relu(output)
        with tf.variable_scope("conv2_224x224"):
            output = self.conv2d_b(output, 64, 3)
            output = self.relu(output)
        with tf.variable_scope("pool1"):
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv3_112x112"):
            output = self.conv2d_b(output, 128, 3)
            output = self.relu(output)
        with tf.variable_scope("conv4_112x112"):
            output = self.conv2d_b(output, 128, 3)
            output = self.relu(output)
        with tf.variable_scope("pool2"):
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv5_56x56"):
            output = self.conv2d_b(output, 256, 3)
            output = self.relu(output)
        with tf.variable_scope("conv6_56x56"):
            output = self.conv2d_b(output, 256, 3)
            output = self.relu(output)
        with tf.variable_scope("conv7_56x56"):
            output = self.conv2d_b(output, 256, 3)
            output = self.relu(output)
        with tf.variable_scope("pool3"):
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv8_28x28"):
            output = self.conv2d_b(output, 512, 3)
            output = self.relu(output)
        with tf.variable_scope("conv9_28x28"):
            output = self.conv2d_b(output, 512, 3)
            output = self.relu(output)
        with tf.variable_scope("conv10_28x28"):
            output = self.conv2d_b(output, 512, 3)
            output = self.relu(output)
        with tf.variable_scope("pool4"):
            output = self.max_pool(output, 2)
        with tf.variable_scope("conv11_14x14"):
            output = self.conv2d_b(output, 512, 3)
            output = self.relu(output)
        with tf.variable_scope("conv12_14x14"):
            output = self.conv2d_b(output, 512, 3)
            output = self.relu(output)
        with tf.variable_scope("conv13_14x14"):
            output = self.conv2d_b(output, 512, 3)
            output = self.relu(output)
        with tf.variable_scope("pool5"):
            output = self.max_pool(output, 2)

        with tf.variable_scope("fc1"):
            output = self.fully_connected(output, 4096)
            output = self.relu(output)
        with tf.variable_scope("dropout1"):
            output = self.dropout(output)
        with tf.variable_scope("fc2"):
            output = self.fully_connected(output, 4096)
            output = self.relu(output)
        with tf.variable_scope("dropout2"):
            output = self.dropout(output)
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
        #correct_prediction = tf.nn.in_top_k(prediction, tf.argmax(self.labels, 1), 5)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
