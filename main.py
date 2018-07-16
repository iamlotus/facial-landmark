import os
from project.model import Model
import tensorflow as tf

def set_environ_for_s3():
    """
    See https://www.tensorflow.org/versions/master/deploy/s3
    :return:
    """
    os.environ['AWS_ACCESS_KEY_ID'] = 'XX'  # Credentials only needed if connecting to a private endpoint
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'XX'
    os.environ['AWS_REGION'] = 'us-west-1' # Region for the S3 bucket, this is not always needed. Default is us-east-1.
    #os.environ['S3_ENDPOINT'] = 's3.us-west-1.amazonaws.com' # The S3 API Endpoint to connect to. This is specified in a HOST:PORT format.amazonaws.com
    # os.environ['S3_USE_HTTPS'] = '1'  # Whether or not to use HTTPS. Disable with 0.
    # os.environ['S3_VERIFY_SSL'] = '1'  # If HTTPS is used, controls if SSL should be enabled. Disable with 0.

    #os.system('env')

if __name__=='__main__':
    set_environ_for_s3()
    demo_file="s3://jinlo-data-north-california/demo.tfrecords"

    input=tf.constant([[0,0,75,80,80],[0,75,80,80,80],[0,75,80,80,80],[0,75,75,80,80],[0,0,0,0,0]],dtype=tf.float32)
    input=tf.reshape(input,[1,5,5,1])
    filter=tf.constant([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=tf.float32)
    filter=tf.reshape(filter,[3,3,1,1])

    c=tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding="VALID")
    with tf.Session() as sess:
        print(sess.run(c))
    # file_name="s3://jinlo-first-backup-bucket/adoc.txt"

    # model=Model([demo_file], [demo_file])
    # model.train()
    # model.test(['./data/tfrecords/demo.tfrecords'])