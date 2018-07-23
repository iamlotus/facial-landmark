import numpy as np
import tensorflow as tf

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
tf.app.flags.DEFINE_integer('train_num_epocs', 10, '''[Train] epoc numbers''')
tf.app.flags.DEFINE_integer('train_steps', 20000, '''[Train] train steps''')

tf.app.flags.DEFINE_string('train_file_path', 'data/tfrecords/train',
                           '''[Train] where to find validate file (in tfrecord format)''')
tf.app.flags.DEFINE_string('model_dir', 'logs/train', '''[Train] where to save checkpoint and tensorboard output''')
tf.app.flags.DEFINE_string('exported_model_dir', '[Train]logs/exported_model',
                           '''[Train] where to save checkpoint and tensorboard output''')
# other
tf.app.flags.DEFINE_string('mode', 'train', '''[GLOBAL] which mode to run, must be 'train'/'eval'/'predict' ''')
tf.app.flags.DEFINE_string('test_file_path', 'data/tfrecords/test',
                           '''[Test] where to find validate file (in tfrecord format)''')
tf.app.flags.DEFINE_string('validate_file_path', 'data/tfrecords/validate',
                           '''[Validate] where to find validate file (in tfrecord format)''')

def cnn(features,labels,mode):
    layers=[]

    """
        The model function for the network.
        """
    # |== Layer 0: input layer ==|
    # Input feature x should be of shape (batch_size, image_width, image_height, color_channels).
    # Image shape should be checked for safety reasons at early stages, and could be removed
    # before training actually starts.
    # assert features['x'].shape[1:] == (
    #     IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), "Image size does not match."
    inputs = tf.to_float(features['x'], name="input_to_float")
    layers.append(inputs)

    # |== Layer 1 ==|

    # Convolutional layer.
    # Computes 32 features using a 3x3 filter with ReLU activation.
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)
    layers.append(conv1)

    # Pooling layer.
    # First max pooling layer with a 2x2 filter and stride of 2.
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid')
    layers.append(pool1)

    # |== Layer 2 ==|

    # Convolutional layer
    # Computes 64 features using a 3x3 filter with ReLU activation.
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)
    layers.append(conv2)

    # Convolutional layer
    # Computes 64 features using a 3x3 filter with ReLU activation.
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)
    layers.append(conv3)

    # Pooling layer
    # Second max pooling layer with a 2x2 filter and stride of 2.
    pool2 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid')
    layers.append(pool2)

    # |== Layer 3 ==|

    # Convolutional layer
    # Computes 64 features using a 3x3 filter with ReLU activation.
    conv4 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)
    layers.append(conv4)

    # Convolutional layer
    # Computes 64 features using a 3x3 filter with ReLU activation.
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)
    layers.append(conv5)

    # Pooling layer
    # Third max pooling layer with a 2x2 filter and stride of 2.
    pool3 = tf.layers.max_pooling2d(
        inputs=conv5,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid')
    layers.append(pool3)

    # |== Layer 4 ==|

    # Convolutional layer
    # Computes 128 features using a 3x3 filter with ReLU activation.
    conv6 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)
    layers.append(conv6)

    # Convolutional layer
    # Conputes 128 features using a 3x3 filter with ReLU activation.
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)
    layers.append(conv7)

    # Pooling layer
    # Fourth max pooling layer with a 2x2 filter and stride of 2.
    pool4 = tf.layers.max_pooling2d(
        inputs=conv7,
        pool_size=[2, 2],
        strides=(1, 1),
        padding='valid')
    layers.append(pool4)

    # |== Layer 5 ==|

    # Convolutional layer
    conv8 = tf.layers.conv2d(
        inputs=pool4,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)
    layers.append(conv8)

    # |== Layer 6 ==|

    # Flatten tensor into a batch of vectors
    flatten = tf.layers.flatten(inputs=conv8)
    layers.append(layers)

    # Dense layer 1, a fully connected layer.
    dense1 = tf.layers.dense(
        inputs=flatten,
        units=1024,
        activation=tf.nn.relu,
        use_bias=True)
    layers.append(dense1)

    # Dense layer 2, also known as the output layer.
    logits = tf.layers.dense(
        inputs=dense1,
        units=136,
        activation=None,
        use_bias=True,
        name="logits")
    layers.append(logits)

    return layers

def model_fn(features, labels, mode):

    layers=cnn(features,labels,mode)
    logits=layers[-1]

    # Make prediction for PREDICATION mode.
    predictions_dict = {
        "name": features['name'],
        "logits": logits
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)

    # Caculate loss using mean squared error.
    label_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
    loss = tf.losses.mean_squared_error(
        labels=label_tensor, predictions=logits)

    # Configure the train OP for TRAIN mode.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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
        "MSE": tf.metrics.root_mean_squared_error(
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

    return {"x": image_reshaped, "name": parsed_features['image/filename']}, points


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
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)
    if cache is True:
        dataset=dataset.cache()

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
        batch_size=2,
        num_epochs=1,
        shuffle=False)


def _predict_input_fn():
    """Function for predicting."""
    return input_fn(
        record_file=FLAGS.test_file_path,
        batch_size=2,
        num_epochs=1,
        shuffle=False)

def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    image = tf.placeholder(dtype=tf.uint8,
                           shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL],
                           name='input_image_tensor')
    receiver_tensor = {'image': image}
    feature = tf.reshape(image, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
    return tf.estimator.export.ServingInputReceiver(feature, receiver_tensor)

def main(unused_argv):
    """MAIN"""
    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.model_dir)

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
        estimator.export_savedmodel(FLAGS.exported_model_dir, serving_input_receiver_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:
        evaluation = estimator.evaluate(input_fn=_eval_input_fn)
        print(evaluation)

    else:
        predictions = estimator.predict(input_fn=_predict_input_fn)
        for _, result in enumerate(predictions):
            img = cv2.imread(result['name'].decode('ascii'))
            marks = np.reshape(result['logits'], (-1, 2)) * IMG_WIDTH
            for mark in marks:
                cv2.circle(img, (int(mark[0]), int(
                    mark[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)
            img = cv2.resize(img, (512, 512))
            cv2.imshow('result', img)
            cv2.waitKey()


if __name__ == '__main__':
    tf.app.run()