import os
from project.model import Model

if __name__=='__main__':
    train_file="data/tfrecords/train.tfrecords"
    validation_file="data/tfrecords/validation.tfrecords"
    test_file="data/tfrecords/test.tfrecords"

    model=Model([train_file], [validation_file])
    model.train()
    model.test(['./data/tfrecords/test.tfrecords'])
