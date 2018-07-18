from project.model import Model
from project.hyper_parameters  import *
import os



if __name__=='__main__':

    # use gpu1 only
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    train_file="data/tfrecords/train.tfrecords"
    validate_file = "data/tfrecords/validate.tfrecords"
    test_file = "data/tfrecords/test.tfrecords"

    test_paths=['data/300VW/011/image/000001.jpg','data/helen/testset/30427236_1.jpg','data/300W/indoor_001.png']

    model=Model([train_file], [validate_file])

    if FLAGS.phase == 'train':
        model.train()
    elif FLAGS.phase == 'test':
        model.test([test_file])
    elif FLAGS.phase == 'test2':
        model.test2(test_paths)
    else:
        raise ValueError("phase must be 'train' or 'test', meet %s"%FLAGS.phases)