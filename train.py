import argparse
import os
import json
import time
from datetime import datetime
from time import gmtime, strftime

from PIL import Image

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import cifar10

import data
import alexnet
import vgg_like

model = vgg_like
parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='cifar-10-batches-py',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--train_dir', type=str, default='./runs',
                    help='Path to the runs directory.')

parser.add_argument('--config_path', type=str, default='./config.json',
                    help='Path to training config json file.')

parser.add_argument('-r', action='store_true',
                    help='Wether or not to continue training on saved model.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--lr', type=float, default=None,
                    help='Training rate overrides.')

def parse_config():
    """
    parse training settings from a json file
    """
    with open(FLAGS.config_path) as config_file:
        config_json = json.load(config_file)

    return config_json

def start_training():
    """Train CIFAR-10 for a number of steps."""
    (training_data, training_labels), (_, _) = cifar10.load_data()
    training_data = training_data.astype(np.float32)
    training_labels = training_labels.reshape(-1).astype(np.int32)

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        with tf.name_scope('input'):
            input_data = tf.constant(training_data)
            input_labels = tf.constant(training_labels)

            image, label = tf.train.slice_input_producer([input_data, input_labels],
                                                         num_epochs=CONFIG["epoch"])
            images, labels = tf.train.batch([image, label], batch_size=CONFIG["batch_size"])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images, CONFIG)

        # Calculate loss.
        loss = model.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss, global_step, CONFIG, FLAGS.lr)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs([loss, global_step])

            def after_run(self, run_context, run_values):
                if self._step % CONFIG["log_frequency"] == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value, global_step_num = run_values.results
                    examples_per_sec = CONFIG["log_frequency"] * CONFIG["batch_size"] / duration
                    sec_per_batch = float(duration / CONFIG["log_frequency"])

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), global_step_num, loss_value,
                            examples_per_sec, sec_per_batch))

        gpu_config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        gpu_config.gpu_options.allow_growth = True
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=CONFIG["epoch"]*data.NUM_TRAINING_RECORD),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=gpu_config,
            save_checkpoint_secs=60
            ) as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
    data.maybe_download_and_extract(FLAGS.data_dir)

    if not FLAGS.r:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    start_training()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    CONFIG = parse_config()
    tf.app.run()
