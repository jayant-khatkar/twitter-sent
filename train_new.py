'''
Train various models and display dev and training performance at the end of each epoch

and display test performance at the end

Jayant Khatkar
'''

import tensorflow as tf
import numpy as np
import os
import data_helper
import sys

#load model(s)
model_locations = os.path.abspath(os.path.curdir)
model_locations = os.path.join(model_locations, "Models")
sys.path.insert(0, model_locations)
from GRU_net import *
from Deep_RNN2 import *
from biRNN import *
from CNN1 import *
from CNN2 import *

from   sklearn.metrics import confusion_matrix, f1_score
from   test import *
from   restorer import *


# Data parameters
embedding_size   = 400
n_words          = 120
n_classes        = 3
local_path       = "/path/to/data/"
train_path       = local_path + "train_set.txt"
dev_path         = local_path + "dev_set.txt"
test_path        = local_path + "stest_set.txt"
num_epochs       = 30
batch_size       = 200

# organise output files to track hyperparameter tuning
run_ID           = 4

#####GRU_net
learning_rate    = [ 0.01,  0.01,  0.01,  0.01,  0.01]
drop_keep_prob   = [  0.9,   0.9,   0.9,     1,     1]
RNN_cells        = [  256,   256,   256,   256,   256]
RNN_layers       = [    1,     1,     1,     1,     1]
words_used       = [   50,    80,    80,    40,    50]
fully_connected  = [  512,   256,   512,   512,   256]

#####biRNN
learning_rate     = [ 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
drop_keep_prob    = [      1,      1,      1,      1,      1]
RNN_cells         = [    256,    256,    256,    256,    256]
RNN_layers        = [      1,      2,      2,      2,      3]
words_used        = [     60,     60,     50,     40,     40]
fully_connected   = [    512,    512,    512,    512,    512]


hyperparams       = [
    learning_rate[run_ID],
    drop_keep_prob[run_ID],
    RNN_cells[run_ID],
    RNN_layers[run_ID],
    words_used[run_ID],
    fully_connected[run_ID]
    ]


#####CNN1
CNN1_hyper0 = [0.0001, 0.7, 64, [1,2,3,4,5], 512]
CNN1_hyper1 = [0.0001, 0.6, 64, [1,2,3,4,5], 512]
CNN1_hyper2 = [0.0001, 0.5, 64, [1,2,3,4,5], 512]

#####CNN2
CNN2_hyper0 = [0.0001, 0.5, 64, [1,2,3,4,5], 512]
CNN2_hyper1 = [0.0001, 0.4, 64, [1,2,3,4,5], 512]
CNN2_hyper2 = [0.0001, 0.3, 64, [1,2,3,4,5], 512]

def train_model():

    x_in  = tf.placeholder(
        "float",
        [None, n_words * embedding_size],
        name="x_in"
        )

    target = tf.placeholder(
        "float",
        [None, n_classes],
        name = "target"
        )

    model = biRNN(
        x_in,
        target,
        embedding_size,
        n_words,
        n_classes,
        hyperparams
        )

    #print model details for the log
    model.info()

    # save directeries
    tensorboard_path, op_names_path, model_path = save_paths(
        model.model_name,
        run_ID
        )


    sess  = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(
        tensorboard_path,
        graph=tf.get_default_graph()
        )

    best_f1=0
    print('\nTraining Model...\n')
    for epoch in range(num_epochs):

        # generate batches
        batches_train = get_zipped_batches(train_path, batch_size)
        batch_index = 0

        #train on batches
    	for train_batch in batches_train:
            train_batch_xs, train_batch_ys = zip(*train_batch)
    	    _ , summary = sess.run(
                [model.optimize, model.tensorboard_summary],
                feed_dict={
                    x_in:   train_batch_xs,
                    target: train_batch_ys
                    }
                )

            #tensorboard summary
            summary_writer.add_summary(summary, epoch*batch_size + batch_index)
            batch_index += 1

        #calculate train & dev performance after training
        trn_acc, trn_f1, trn_cm = test_model(train_path, model, sess)
        dev_acc, dev_f1, dev_cm = test_model(dev_path,   model, sess)
        display_epoch_performance(epoch, dev_acc, dev_f1, dev_cm, "dev")
        display_epoch_performance(epoch, trn_acc, trn_f1, trn_cm, "trn")

    	#if improved performance, then save model
    	if best_f1 < dev_f1:
    	    best_f1 = dev_f1
            _ = saver.save(sess, model_path)

    #test acter completing training
    test_acc, test_f1, test_cm = test_model(test_path, model, sess)
    display_epoch_performance("final", test_acc, test_f1, test_cm, "test")

if __name__ == '__main__':
  train_model()
