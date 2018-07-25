from tensorflow.python.estimator.canned.baseline import *
import tensorflow as tf


classifier=BaselineClassifier(n_classes=3)

def input_fn_train():
    # returns x, y (where y represents label's class index).
    x= tf.constant([[1.,2.],[3.,4.],[5.,6.]],dtype=tf.float32)
    y=tf.constant([0,1,2],dtype=tf.float32)
    return x,y

def input_fn_eval():# returns x, y (where y represents label's class index).
    x = tf.constant([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]],dtype=tf.float32)
    y = tf.constant([0, 1, 2], dtype=tf.float32)
    return x, y


#Fit model
classifier.train(input_fn=input_fn_train)


# Evaluate cross entropy between the test and train lables

eval_result=classifier.evaluate(input_fn=input_fn_eval)
loss=eval_result["loss"]


new_samples=[[1.,2.,3.]]
predictions= classifier.predict(new_samples)