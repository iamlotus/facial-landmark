# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## remain this line to run in jupyter notebook
## what this essentially does is that it creates a flag f as jupyter notebook supplies a flag "-f"
# to pass a name for a JSON file likely for the kernel.
tf.app.flags.DEFINE_string('f', '', 'kernel')


## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_integer('report_freq', 391, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_integer('dot_freq', 1, '''Steps taks to print dot in console''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


## The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_steps', 80000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_integer('train_batch_size', 64, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 128, '''Validation batch size''')
tf.app.flags.DEFINE_integer('test_batch_size', 10, '''Test batch size''')

tf.app.flags.DEFINE_integer('landmark_point_num', 68, '''Landmark point number''')

tf.app.flags.DEFINE_float('init_lr', 0.1, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each
time''')
tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 60000, '''At which step to decay the learning rate''')


## The following flags define hyper-parameters modifying the training network
tf.app.flags.DEFINE_integer('save_checkpoint_intervals', 10000, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')

tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue
training''')

logs_dir= 'logs'
train_dir = logs_dir+'/logs_' + str(FLAGS.num_residual_blocks)+'_blocks' + '/'
ckpt_dir=logs_dir+'/ckpts_'+str(FLAGS.num_residual_blocks)+'_blocks'
ckpt_file=ckpt_dir+'/model.ckpt'