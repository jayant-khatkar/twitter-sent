"""
Test model
"""

import tensorflow as tf
import numpy as np
import data_helper
import os
from   sklearn.metrics import confusion_matrix, f1_score

def get_zipped_batches(file_path, batch_size):
    """
    funciton to fill in constant variables
    """
    epochs     = 1
    label_last = False
    header     = False
    delimiter  = " "
    labels     = ['positive', 'negative', 'objective']

    batches = data_helper.batch_iter_gen(
        file_path,
        batch_size,
        epochs,
        label_last,
        header,
        delimiter,
        labels
        )

    return batches

def test_model_internal(file_path, sess, placeholders, measure_performance):
    """
    Reads data at file_path
    
    Internal function is separate so it can be accessed by restored model (when the class is not known)

    Returns accuracy, average f1 score and confusion matrix
    """
    batch_size = 200
    batches_test = get_zipped_batches(file_path, batch_size)

    accuracy  = np.array([])
    y_pred    = np.array([])
    y_real    = np.array([])

    for test_batch in batches_test:

        test_batch_xs, test_batch_ys = zip(*test_batch)

        accuracy_batch, y_pred_batch, y_real_batch = sess.run(
            measure_performance,
            feed_dict = {
                placeholders[0]:   test_batch_xs,
                placeholders[1]:   test_batch_ys
                }
            )

        y_pred   = np.append(y_pred,   y_pred_batch)
        y_real   = np.append(y_real,   y_real_batch)
        accuracy = np.append(accuracy, accuracy_batch)

    accuracy = accuracy.mean()
    labels = [0,1,2]
    f1_sc = f1_score(y_real, y_pred, average=None, labels = labels)
    pos_neg_f1 = (f1_sc[0]+f1_sc[1])/2

    cm = confusion_matrix(y_real, y_pred, labels)

    return accuracy, pos_neg_f1, cm

def test_model(file_path, model, sess):
    """
    wrapper for the internal function so it can easily accessed when the model is given

    Returns accuracy, average f1 score and confusion matrix
    """
    placeholders = [model.x_in, model.y]
    return test_model_internal(file_path, sess, placeholders, model.measure_performance)

def display_epoch_performance(epoch, accuracy, f1, confusion_matrix, source):
    print('At Epoch: '+ str(epoch))
    print('    ' + source + '    accuracy: '   + str(accuracy))
    print('    ' + source + '    f1_score: '   + str(f1))
    print('    ' + source + ' conf matrix: \n' + str(confusion_matrix) +'\n')


def save_paths(model_name, run_ID):
    path = os.path.abspath(os.path.curdir)
    path = os.path.join(path, model_name)
    path = os.path.join(path, str(run_ID))
    tensorboard = os.path.join(path, "tensorboard_log")
    name_save   = os.path.join(path, "op_names.txt")
    model_save  = os.path.join(path, "model_save")


    if not os.path.exists(tensorboard):
        os.makedirs(tensorboard)

    if not os.path.exists(model_save):
        os.makedirs(model_save)

    model_save  = os.path.join(model_save,"model.ckpt")
    
    return tensorboard, name_save, model_save
