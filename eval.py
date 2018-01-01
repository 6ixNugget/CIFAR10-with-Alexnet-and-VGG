from datetime import datetime
import math
import time
import os
import json
from time import gmtime, strftime

import numpy as np
import tensorflow as tf

import train
import data
import alexnet
import vgg_like

model = vgg_like
parser = train.parser

parser.add_argument('--eval_dir', type=str, default='./evals',
                    help='Directory where to write event logs.')

parser.add_argument('--checkpoint_dir', type=str, default='./runs',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=60,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=10000,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=False,
                    help='Whether to run eval only once.')

def parse_config():
    """
    parse training settings from a json file
    """
    with open(FLAGS.config_path) as config_file:
        config_json = json.load(config_file)

    return config_json

def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                start=True))

            num_iter = int(math.ceil(data.NUM_TEST_RECORD / CONFIG["batch_size"]))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * CONFIG["batch_size"]
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    test_data, test_labels = data.read_in_test_data(FLAGS.data_dir)
    
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.

            with tf.name_scope('input'):
                input_data = tf.constant(test_data)
                input_labels = tf.constant(test_labels)

                image, label = tf.train.slice_input_producer([input_data, input_labels])
                label = tf.cast(label, tf.int32)
                images, labels = tf.train.batch([image, label], batch_size=CONFIG["batch_size"])

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = model.inference(images, CONFIG)

            # Calculate predictions.
            top_k_op = tf.nn.in_top_k(logits, labels, 1)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                CONFIG["moving_average_decay"])
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

            while True:
                eval_once(saver, summary_writer, top_k_op, summary_op)
                if FLAGS.run_once:
                    break
                time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    data.maybe_download_and_extract(FLAGS.data_dir)
    current_eval_dir = os.path.join(FLAGS.eval_dir, "eval" + strftime("_%m%d_%H%M", gmtime()))
    if tf.gfile.Exists(current_eval_dir):
        tf.gfile.DeleteRecursively(current_eval_dir)
    tf.gfile.MakeDirs(current_eval_dir)
    evaluate()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    CONFIG = parse_config()
    tf.app.run()