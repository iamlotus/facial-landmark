import os
from project.model import Model

if __name__=='__main__':
    train_file="data/tfrecords/train.tfrecords"
    validate_file="data/tfrecords/validate.tfrecords"
    test_file="data/tfrecords/test.tfrecords"

    model=Model([train_file], [validate_file])
    model.train()
    model.test([test_file])
