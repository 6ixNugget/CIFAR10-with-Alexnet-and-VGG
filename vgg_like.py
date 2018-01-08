import tensorflow as tf

import data

def _get_variable_with_normal_init(name, shape, stddev):
    v = tf.get_variable(name,
                        shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    return v

def inference(images, CONFIG):

    weight_decay = CONFIG["weight_decay"]

    # Block 1: 2 conv + 1 pool, input 32 * 32
    conv1_1 = _conv_layer(images, 3, 64, weight_decay, "conv1_1")
    conv1_1_dropout = tf.nn.dropout(conv1_1, 0.3)
    conv1_2 = _conv_layer(conv1_1_dropout, 64, 64, weight_decay, "conv1_2")
    pool1 = _max_pool(conv1_2, "pool1")

    # Block 2: 2 conv + 1 pool, input 16 * 16
    conv2_1 = _conv_layer(pool1, 64, 128, weight_decay, "conv2_1")
    conv2_1_dropout = tf.nn.dropout(conv2_1, 0.4)
    conv2_2 = _conv_layer(conv2_1_dropout, 128, 128, weight_decay, "conv2_2")
    pool2 = _max_pool(conv2_2, "pool2")

    # Block 3: 3 conv + 1 pool, input 8 * 8
    conv3_1 = _conv_layer(pool2, 128, 256, weight_decay, "conv3_1")
    conv3_1_dropout = tf.nn.dropout(conv3_1, 0.4)
    conv3_2 = _conv_layer(conv3_1_dropout, 256, 256, weight_decay, "conv3_2")
    conv3_2_dropout = tf.nn.dropout(conv3_2, 0.4)
    conv3_3 = _conv_layer(conv3_2_dropout, 256, 256, weight_decay, "conv3_3")
    pool3 = _max_pool(conv3_3, "pool3")

    # Block 4: 3 conv + 1 pool, input 4 * 4
    conv4_1 = _conv_layer(pool3, 256, 512, weight_decay, "conv4_1")
    conv4_1_dropout = tf.nn.dropout(conv4_1, 0.4)
    conv4_2 = _conv_layer(conv4_1_dropout, 512, 512, weight_decay, "conv4_2")
    conv4_2_dropout = tf.nn.dropout(conv4_2, 0.4)
    conv4_3 = _conv_layer(conv4_2_dropout, 512, 512, weight_decay, "conv4_3")
    pool4 = _max_pool(conv4_3, "pool4")

    # Block 5: 3 conv + 1 pool, input 2 * 2
    conv5_1 = _conv_layer(pool4, 512, 512, weight_decay, "conv5_1")
    conv5_1_dropout = tf.nn.dropout(conv5_1, 0.4)
    conv5_2 = _conv_layer(conv5_1_dropout, 512, 512, weight_decay, "conv5_2")
    conv5_2_dropout = tf.nn.dropout(conv5_2, 0.4)
    conv5_3 = _conv_layer(conv5_2_dropout, 512, 512, weight_decay, "conv5_3")
    pool5 = _max_pool(conv5_3, "pool5")

    # FC: input 1 * 1
    fc5 = _fc_layer_with_activation(pool5, 512, 512, tf.nn.relu, "fc5")
    fc5_dropout = tf.nn.dropout(fc5, CONFIG["dropout"])
    fc6 = _fc_layer(fc5_dropout, 512, data.NUM_CLASSES, "fc6")

    return fc6


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, global_step, CONFIG, lr_overrides=None):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = data.NUM_TRAINING_RECORD / CONFIG["batch_size"]
    decay_steps = int(num_batches_per_epoch * CONFIG["num_epoch_per_decay"])

    # Decay the learning rate exponentially based on the number of steps.
    if lr_overrides:
        lr = lr_overrides
    else:
        lr = tf.train.exponential_decay(CONFIG['initial_lr'],
                                        global_step,
                                        decay_steps,
                                        CONFIG['lr_decay_factor'],
                                        staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        CONFIG["moving_average_decay"], global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
    tf.nn.zero_fraction(x))

def _max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def _conv_layer(bottom, in_channels, out_channels, weight_decay, name):
    with tf.variable_scope(name):
        filt, bias = _get_conv_var(3, in_channels, out_channels, weight_decay, name)

        conv_output = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias_output = tf.nn.bias_add(conv_output, bias)
        batch_norm_output = tf.layers.batch_normalization(bias_output, training=True)
        relu = tf.nn.relu(batch_norm_output)

        _activation_summary(relu)

        return relu

def _fc_layer(bottom, in_size, out_size, name):
    with tf.variable_scope(name):
        weights, biases = _get_fc_var(in_size, out_size, name)

        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    return fc

def _fc_layer_with_activation(bottom, in_size, out_size, activation, name):
    fc = _fc_layer(bottom, in_size, out_size, name)
    act = activation(fc)
        
    _activation_summary(act)

    return act

def _get_conv_var(filter_size, in_channels, out_channels, weight_decay, name):
    #initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    filt = tf.get_variable(name + "_filter", shape=[filter_size, filter_size, in_channels, out_channels],
                           initializer=tf.contrib.layers.xavier_initializer())

    if weight_decay is not None:
        wd = tf.multiply(tf.nn.l2_loss(filt), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', wd)

    initial_value = tf.truncated_normal([out_channels], .0, .001)
    bias = tf.Variable(initial_value, name=name + "_bias")

    return filt, bias

def _get_fc_var(in_size, out_size, name):
    # initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
    # filt = tf.Variable(initial_value, name=name + "_filter")

    filt = tf.get_variable(name + "_filter", shape=[in_size, out_size],
                           initializer=tf.contrib.layers.xavier_initializer())
    initial_value = tf.truncated_normal([out_size], .0, .001)
    bias = tf.Variable(initial_value, name=name + "_bias")

    return filt, bias