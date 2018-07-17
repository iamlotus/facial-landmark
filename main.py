from project.model import Model
from project.hyper_parameters  import *



if __name__=='__main__':

    train_file="data/tfrecords/train.tfrecords"
    validate_file = "data/tfrecords/validate.tfrecords"
    test_file = "data/tfrecords/test.tfrecords"

    model=Model([train_file], [validate_file])

    if FLAGS.phase == 'train':
        model.train()
    elif FLAGS.phase == 'test':
        model.test([test_file])
    else:
        raise ValueError("phase must be 'train' or 'test', meet %s"%FLAGS.phases)