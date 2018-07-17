import numpy as np
import os
from project import *
from project.prepare_data import decode_from_tfrecords,find_path_from_url
from project.hyper_parameters import *
from project.resnet import inference
from project.face_draw import draw_landmarks
from datetime import datetime
from time import time
from tensorflow.python.client import timeline
import cv2

class entry():
    def __init__(self,name):
        self.name=name

    def __enter__(self):
        self.start=datetime.now()
        print('【Start】%s'%self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end=datetime.now()
        duration=end-self.start
        print('【End】%s in %d seconds'%(self.name,duration.seconds))

class Model(object):
    def __init__(self,train_file_names,
                 validate_file_names,
                 img_size=(128,128,3)):

        self._img_size=img_size
        self.prepare_data(train_file_names,validate_file_names)
        self.placeholders()

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running merge_summary_op. Initialize a new session
        self.saver = tf.train.Saver()

        init = tf.global_variables_initializer()

        self.sess=tf.Session()

        print('Prepare MODEL')

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print('create dir %s'%logs_dir)

        if FLAGS.is_use_ckpt is True or FLAGS.phase=='test':
            last_ckpt=tf.train.latest_checkpoint(ckpt_dir)
            with entry('Restore variables from %s'%last_ckpt):
                if not last_ckpt:
                    raise RuntimeError('can not find any ckpt from %s while is_use_ckpt==True'%ckpt_dir)
                self.saver.restore(self.sess,last_ckpt)
        else:
            with entry('Init variables'):
                self.sess.run(init)

            if os.path.exists(train_dir):
                with entry('Clean %s'%train_dir):
                    for the_file in os.listdir(train_dir):
                        file_path = os.path.join(train_dir, the_file)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)


    def prepare_data(self,train_file_names,validate_file_names):
        self.train_image, self.train_pts, self.train_file=self.extract_data(train_file_names,FLAGS.train_batch_size,shuffle=False)
        self.train_image=tf.cast(self.train_image,tf.float32)

        self.validate_image, self.validate_pts, self.validate_file=self.extract_data(validate_file_names,FLAGS.validation_batch_size,shuffle=False)
        self.validate_image = tf.cast(self.validate_image, tf.float32)

    @staticmethod
    def extract_data(file_names, batch_size,shuffle):
        file_name_queue = tf.train.string_input_producer(file_names)
        image, pts, file = decode_from_tfrecords(file_name_queue,batch_size=batch_size,shuffle=shuffle)
        return image,pts,file

    def placeholders(self):
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[],name='lr_placeholder')
        self.test_image=tf.placeholder(dtype=tf.float32,shape=[None,*self._img_size],name='test_image')


    def build_train_validation_graph(self):
        """
        This function builds the train graph and validation graph at the same time.

        """

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.train_image, FLAGS.num_residual_blocks, reuse=False)
        vali_logits = inference(self.validate_image, FLAGS.num_residual_blocks, reuse=True)
        self.global_step = tf.Variable(-1,name='global_step',trainable=False)

        # The following codes calculate the train loss, which is consist of the
        # mse and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.train_pts)
        #self.train_loss = tf.add_n([loss] + regu_losses)
        self.train_loss = loss

        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.validate_pts)
        # self.train_op = self.train_operation(self.train_loss)
        self.train_op = self.train_operation(loss)
        self.vali_op = self.validation_operation(self.vali_loss)


    def build_test_graph(self):
        test_logits = inference(self.test_image, FLAGS.num_residual_blocks, reuse=True)
        self.test_op = test_logits

    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and labels
        :param logits: 2D tensor with shape [batch_size, num_pts]
        :param labels: 2D tensor with shape [batch_size, num_pts]
        :return: loss tensor with shape [1]
        '''
        mse = tf.losses.mean_squared_error(labels,logits)
        return mse

    def train_operation(self, total_loss):
        '''
        Defines train operations
        :param total_loss: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once.
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op =opt.minimize(total_loss)
        return train_op

    def validation_operation(self, val_loss):
        '''
        Defines validation operations
        :param loss: tensor with shape [1]
        :return: validation operation
        '''
        tf.summary.scalar('val_loss', val_loss)
        return tf.identity(val_loss)

    def train(self):
        """
        Continue training
        :return:
        """
        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, self.sess.graph)
        merge_summary_op = tf.summary.merge_all()
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord,sess=self.sess)  # 启动QueueRunner, 此时文件名队列已经进队。
        start_time = time()

        # if there is no checkpoint, global_step starts from -1, else global_step starts from an *FINISHED* step
        # in both scenarios, start from an new step (+1)
        start=self.global_step.eval(session=self.sess)+1

        for step in range(start, FLAGS.train_steps):
            self.global_step.assign(step).eval(session=self.sess)

            if step % FLAGS.report_freq == 0:
                # trace train/validate loss
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary_str,_,_,train_loss_value,validate_loss_value=self.sess.run([merge_summary_op, self.train_op, self.vali_op, self.train_loss, self.vali_loss], options=options, run_metadata=run_metadata, feed_dict={self.lr_placeholder: FLAGS.init_lr})
                summary_writer.add_run_metadata(run_metadata,'step%05d'%step)
                summary_writer.add_summary(summary_str,global_step=step)
                summary_writer.flush()

                duration = time() - start_time

                format_str = "\n%s: step %d, train loss= %.8f, validate loss = %.8f, %.4f seconds elapsed"
                print(format_str % (datetime.now(), step,train_loss_value, validate_loss_value, duration))

            elif step % FLAGS.report_freq == 1:
                # trace pure train
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary_str,_=self.sess.run([merge_summary_op,self.train_op],
                              feed_dict={self.lr_placeholder: FLAGS.init_lr},options=options, run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata, 'step%05d' % step)
                summary_writer.add_summary(summary_str, global_step=step)
                summary_writer.flush()

                # write json
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(train_dir+'timeline_%d.json'%step, 'w') as f:
                    f.write(chrome_trace)
            else:
                self.sess.run(self.train_op,
                        feed_dict={self.lr_placeholder: FLAGS.init_lr})

            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print ('Learning rate decayed to ', FLAGS.init_lr)

            # Save checkpoints
            if (step>0 and step % FLAGS.save_checkpoint_intervals == 0) or (step + 1 == FLAGS.train_steps):
                self.saver.save(self.sess, ckpt_file, global_step=self.global_step)
                print('save check point to '+ckpt_file)



            if step%(FLAGS.dot_freq*100)==0:
                print()
            elif step%FLAGS.dot_freq==0:
                print('.',end='',sep='',flush=True)

        coord.request_stop()
        coord.join(threads)

    def test(self,test_file_names=['../data/tfrecords/test.tfrecords'], show_original_pts=False):
        self.build_test_graph()

        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程

        image, pts, file=self.extract_data(test_file_names,FLAGS.test_batch_size,shuffle=True)
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)  # 启动QueueRunner, 此时文件名

        images,original_pts,f=self.sess.run([image,pts,file])
        output_pts= self.sess.run(self.test_op,feed_dict={self.test_image:images})

        n=20
        for i in range(n):

            img,output_pt,original_pt,url=images[i],output_pts[i],original_pts[i],f[i]
            # img, opts, url = images[i], ps[i], f[i]
            output_pt=np.reshape(output_pt,[-1,2])

            url=url.decode()

            # path=find_path_from_url(url)
            # img = cv2.imread(path)

            draw_landmarks(img,output_pt,1)



            if FLAGS.show_original_pts_in_test:
                original_pt = np.reshape(original_pt, [-1, 2])
                draw_landmarks(img,original_pt,1,color=MARK_COLOR_RED)

            # create a named window and move it
            cv2.namedWindow(url)
            cv2.moveWindow(url, 300, 300)

            cv2.imshow(url, img)
            print('test %d/%d'%(i+1,n))
            cv2.waitKey(0)
            cv2.destroyWindow(url)
        coord.request_stop()
        coord.join(threads)

