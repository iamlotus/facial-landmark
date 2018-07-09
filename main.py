import os
from project.model import Model
from tensorflow.python.lib.io import file_io



def set_environ_for_s3():
    """
    See https://www.tensorflow.org/versions/master/deploy/s3
    :return:
    """
    os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAI4DO5NJJ2EQGZAXQ'  # Credentials only needed if connecting to a private endpoint
    os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
    os.environ['AWS_REGION'] = 'us-west-1' # Region for the S3 bucket, this is not always needed. Default is us-east-1.
    #os.environ['S3_ENDPOINT'] = 's3.us-west-1.amazonaws.com' # The S3 API Endpoint to connect to. This is specified in a HOST:PORT format.amazonaws.com
    # os.environ['S3_USE_HTTPS'] = '1'  # Whether or not to use HTTPS. Disable with 0.
    # os.environ['S3_VERIFY_SSL'] = '1'  # If HTTPS is used, controls if SSL should be enabled. Disable with 0.

    #os.system('env')

if __name__=='__main__':
    set_environ_for_s3()
    print(os.environ)

    # file_name="s3://jinlo-data-north-california/demo.tfrecords"
    file_name="s3://jinlo-first-backup-bucket/adoc.txt"
    print(file_io.stat(file_name))

    # model=Model(['./data/tfrecords/demo.tfrecords'], ['./data/tfrecords/demo.tfrecords'])
    # model.train()
    # model.test(['./data/tfrecords/demo.tfrecords'])