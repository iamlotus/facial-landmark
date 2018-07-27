import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
import tensorflow.contrib.slim as slim

import os

import cv2

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3

tf.logging.set_verbosity(tf.logging.INFO)



FLAGS = tf.app.flags.FLAGS

# data api
tf.app.flags.DEFINE_integer('prefetch_buffer_size', 256, '''[Data api] prefetch buffer size''')
tf.app.flags.DEFINE_integer('num_parallel_calls', 4, '''[Data api] num parallel calls in mapping''')
# train
tf.app.flags.DEFINE_integer('train_batch_size', 64, '''[Train] batch size''')
tf.app.flags.DEFINE_integer('train_num_epocs', 100, '''[Train] epoc numbers''')
tf.app.flags.DEFINE_integer('train_steps', 500000, '''[Train] train steps''')
tf.app.flags.DEFINE_string('optimizer', 'Adam', '''[Train] optimizer must be 'Adam'/'Adagrad'/'Momentum'/'Sgd'/ftrl' ''')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '''[Train] learning rate ''')
tf.app.flags.DEFINE_string('cuda_visible_devices', '3', '''[Train] visible GPU ''')
tf.app.flags.DEFINE_integer('save_checkpoints_secs', 1200, '''Save checkpoint intervals (in seconds)''')



tf.app.flags.DEFINE_string('train_file_path', 'data/tfrecords/train',
                           '''[Train] where to find validate file (in tfrecord format)''')
tf.app.flags.DEFINE_string('model_dir', 'logs/train', '''[Train] where to save checkpoint and tensorboard output''')
tf.app.flags.DEFINE_string('exported_model_dir', 'logs/exported_model',
                           '''[Train] where to save checkpoint and tensorboard output''')
# other
tf.app.flags.DEFINE_string('mode', 'train', '''[GLOBAL] which mode to run, must be 'train'/'eval'/'predict' ''')
tf.app.flags.DEFINE_string('network', 'cnn', '''[GLOBAL] network, must be 'cnn'/'rnn' ''')
tf.app.flags.DEFINE_string('test_file_path', 'data/tfrecords/test',
                           '''[Test] where to find validate file (in tfrecord format)''')
tf.app.flags.DEFINE_string('validate_file_path', 'data/tfrecords/validate',
                           '''[Validate] where to find validate file (in tfrecord format)''')


model_dir=FLAGS.model_dir+"_"+FLAGS.network
exported_model_dir=FLAGS.exported_model_dir+"_"+FLAGS.network


def rnn(features,mode):

    resnet_image_size=224

    """
        rnn Implementation with slim
        :param features:
        :param mode:
        :return:
    """

    # |== Layer 0: input layer ==|
    # Input feature x should be of shape (batch_size, image_width, image_height, color_channels).
    # Image shape should be checked for safety reasons at early stages, and could be removed
    # before training actually starts.
    assert features['x'].shape[1:] == (
        IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), "Image size does not match."

    images = tf.to_float(features['x'], name="input_to_float")

    inputs = tf.image.resize_images(images, [resnet_image_size, resnet_image_size])

    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training=True
    elif mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:
        is_training = False
    else:
        raise ValueError('unknown mode %s'%mode)

    net, _= resnet_v2.resnet_v2_50(inputs,num_classes=None,is_training=is_training,global_pool=False,reuse=tf.AUTO_REUSE)

    with tf.variable_scope("fc"):
        net = slim.flatten(net)
        net = slim.fully_connected(net, num_outputs=1024, activation_fn=tf.nn.relu)
        logits = slim.fully_connected(net, num_outputs=136, activation_fn=tf.nn.sigmoid)

    return logits


def cnn(features,mode):
    """
    cnn Implementation with slim
    :param features:
    :param mode:
    :return:
    """

    # |== Layer 0: input layer ==|
    # Input feature x should be of shape (batch_size, image_width, image_height, color_channels).
    # Image shape should be checked for safety reasons at early stages, and could be removed
    # before training actually starts.
    assert features['x'].shape[1:] == (
        IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), "Image size does not match."

    inputs = tf.to_float(features['x'], name="input_to_float")

    with slim.arg_scope([slim.conv2d],kernel_size=[3,3],stride=1,padding='VALID',activation_fn=tf.nn.relu), \
            slim.arg_scope([slim.max_pool2d],kernel_size=[2,2],padding='VALID'):

        with tf.variable_scope("layer1"):
            net = slim.conv2d(inputs,num_outputs=32)
            net = slim.max_pool2d(net,stride =2)

        with tf.variable_scope("layer2"):
            net = slim.conv2d(net,num_outputs=64)
            net = slim.conv2d(net, num_outputs=64)
            net = slim.max_pool2d(net,stride =2)

        with tf.variable_scope("layer3"):
            net = slim.conv2d(net, num_outputs=64)
            net = slim.conv2d(net, num_outputs=64)
            net = slim.max_pool2d(net,stride =2)

        with tf.variable_scope("layer4"):
            net = slim.conv2d(net, num_outputs=128)
            net = slim.conv2d(net, num_outputs=128)
            net = slim.max_pool2d(net,stride=2)

        with tf.variable_scope("layer5"):
            net = slim.conv2d(net, num_outputs=256)

        with tf.variable_scope("layer6"):
            net =slim.flatten(net)
            net =slim.fully_connected(net,num_outputs=1024,activation_fn=tf.nn.relu)
            net = slim.fully_connected(net, num_outputs=136, activation_fn=tf.nn.sigmoid)

        return net

def get_optimizer():
    learning_rate=FLAGS.learning_rate
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-6)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif FLAGS.optimizer == 'Sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError("unknown optimizer %s"%FLAGS.optimizer)
    return optimizer


def get_network():

    if FLAGS.network == 'cnn':
        return cnn
    elif FLAGS.network=='rnn':
        return rnn
    else:
        raise ValueError('unknown network %s'%FLAGS.network)


def model_fn(features, labels, mode):
    network=get_network()
    logits=network(features,mode)

    if mode == tf.estimator.ModeKeys.PREDICT:

        # Make prediction for PREDICATION mode.
        predictions_dict = {
            "name": features['name'],
            "logits": logits
        }

        export_outputs_dict ={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:tf.estimator.export.PredictOutput(logits)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict,export_outputs=export_outputs_dict)

    # Calculate loss using mean squared error.
    label_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
    loss = tf.losses.mean_squared_error(
        labels=label_tensor, predictions=logits)

    # Configure the train OP for TRAIN mode.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            export_outputs={'marks': tf.estimator.export.RegressionOutput(logits)})

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "MSE": tf.metrics.mean_squared_error(
            labels=label_tensor,
            predictions=logits)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def _parse_function(record):
    """
    Extract data from a `tf.Example` protocol buffer.
    """
    # Defaults are not specified since both keys are required.
    keys_to_features = {
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/filename2': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'label/points': tf.FixedLenFeature([136], tf.float32)
    }

    parsed_features = tf.parse_single_example(record, keys_to_features)

    # Extract features from single example
    # image_decoded = tf.image.decode_image(parsed_features['image/encoded'])

    image_decoded = tf.decode_raw(parsed_features['image/encoded'], tf.uint8)

    image_reshaped = tf.reshape(
        image_decoded, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
    points = tf.cast(parsed_features['label/points'], tf.float32)

    return {"x": image_reshaped, "name": parsed_features['image/filename2']}, points


def input_fn(record_file, batch_size, num_epochs=None, shuffle=False, cache=True):
    """
    Input function required for TensorFlow Estimator.
    """
    dataset = tf.data.TFRecordDataset(record_file)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(_parse_function,FLAGS.num_parallel_calls)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10000)
    if batch_size != 1:
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(FLAGS.prefetch_buffer_size)
    # cache should be before repeat
    if cache is True:
        dataset=dataset.cache()
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    feature, label = iterator.get_next()
    return feature, label


def _train_input_fn():
    """Function for training."""
    return input_fn(
        record_file=FLAGS.train_file_path,
        batch_size=FLAGS.train_batch_size,
        num_epochs=FLAGS.train_num_epocs,
        shuffle=False)


def _eval_input_fn():
    """Function for evaluating."""
    return input_fn(
        record_file=FLAGS.validate_file_path,
        batch_size=64,
        num_epochs=1,
        shuffle=False)


def _predict_input_fn():
    """Function for predicting."""
    return input_fn(
        record_file=FLAGS.test_file_path,
        batch_size=64,
        num_epochs=1,
        shuffle=True)

def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    image = tf.placeholder(dtype=tf.uint8,
                           shape=[None,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL],
                           name='input_image_tensor')
    name = tf.placeholder(dtype=tf.string,
                           name='input_image_name')

    receiver_tensor = {'x': image}
    feature = {'x':image,'name':name}
    return tf.estimator.export.ServingInputReceiver(feature, receiver_tensor)

def main(unused_argv):
    """MAIN"""

    config = tf.estimator.RunConfig(
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=10,  # Retain the 10 most recent checkpoints.
        log_step_count_steps=100, # log every 500 steps
    )

    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=config)

    # Choose mode between Train, Evaluate and Predict
    mode_dict = {
        'train': tf.estimator.ModeKeys.TRAIN,
        'eval': tf.estimator.ModeKeys.EVAL,
        'predict': tf.estimator.ModeKeys.PREDICT
    }

    mode = FLAGS.mode

    if not mode in mode_dict:
        raise ValueError("argument mode '%s' is not valid, must be one of train/test/predict"%mode)

    mode = mode_dict[mode]

    if mode == tf.estimator.ModeKeys.TRAIN:
        estimator.train(input_fn=_train_input_fn, steps=FLAGS.train_steps)

        # Export result as SavedModel.
        estimator.export_savedmodel(exported_model_dir, serving_input_receiver_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:
        evaluation = estimator.evaluate(input_fn=_eval_input_fn)
        print(evaluation)

    elif mode==tf.estimator.ModeKeys.PREDICT:
        predictions = estimator.predict(input_fn=_predict_input_fn)
        for _, result in enumerate(predictions):
            img = cv2.imread("data/output/"+result['name'].decode('ascii'))
            marks = np.reshape(result['logits'], (-1, 2)) * IMG_WIDTH
            for mark in marks:
                cv2.circle(img, (int(mark[0]), int(
                    mark[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)
            # img = cv2.resize(img, (512, 512))
            cv2.imshow('result', img)
            cv2.waitKey()
    else:
        raise ValueError('unknown mode %s'%mode)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.cuda_visible_devices # set GPU visibility in multiple-GPU environment
    tf.app.run()