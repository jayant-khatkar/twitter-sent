"""
CNN with one fully connected layer Model

applies several filters in one layer (list filtersizes in hyperPs)
maxpools over all words, (essentially picking the best match
for each filter channel across all the words and only using that)
concatantes the different filter ouptus -> dense hidden layer -> output layer

Author: Jayant Khatkar
"""

import tensorflow as tf
import numpy as np
import os
from scope_decorator import *

class CNN1:

    def __init__(self, x_in, target, embed_len, n_words, n_classes, hyperPs):
        """
        x_in and target must be of type tf.placeholder
        remaining arguments are int data parameters
        """
        self.x_in = x_in
        self.x_image = tf.reshape(self.x_in, [-1,n_words,embed_len,1])
        self.y = target

        self.n_classes = n_classes
        self.n_words = n_words
        self.embed_len = embed_len
        self.model_name = "CNN1"

        self.load_hyperparameters(hyperPs)
        self.init_weights
        self.prediction
        self.optimize
        self.measure_performance


    def load_hyperparameters(self, hyperPs):
        self.learning_rate    = hyperPs[0]
        self.drop_keep_prob   = hyperPs[1]
        self.n_channels       = hyperPs[2]
        self.filter_sizes     = hyperPs[3]
        self.hidden_layer     = hyperPs[4]


    @define_scope
    def init_weights(self):
		
        def weight_variable(name, shape):
			W = tf.get_variable(
				name, 
				shape,
				initializer = tf.contrib.layers.xavier_initializer()
				)
			return W
		
        def bias_variable(shape):
			initial = tf.Variable(tf.zeros(shape))
			return initial
        
        self.weights = {}
        self.biases  = {}
		
		# initialize conv filters
        for i in range(len(self.filter_sizes)):
			
			self.weights["filter"+ str(i)] = weight_variable(
				"W_conv_"+str(i),
				[self.filter_sizes[i], 
					self.embed_len,
					1,
					self.n_channels]
				)
				
			self.biases["filter" + str(i)] = bias_variable([self.n_channels])
			
			
			
		# initialize dense layers
        self.weights['dense1'] = weight_variable(
			"dense1", 
			[len(self.filter_sizes)* self.n_channels, self.hidden_layer]
			)
		
        self.weights['dense2'] = weight_variable(
			"dense2", 
			[self.hidden_layer, self.n_classes]
			)

        self.biases['dense1'] = bias_variable([self.hidden_layer])
        self.biases['dense2'] = bias_variable([self.n_classes])


    @define_scope
    def prediction(self):
	
        def conv2d(x, W):
		    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


        def max_pool_2x2(x, kernel_width, kernel_height):
		    return tf.nn.max_pool(x, ksize=[1, kernel_width, kernel_height, 1],
								strides=[1, 1, 1, 1], padding='VALID')
								
		
        #applies different filters to the one input
        convolution_outputs = []
        for i in range(len(self.filter_sizes)):
		
            filter = self.filter_sizes[i]
			
            with tf.variable_scope("filter_"+ str(filter)):
				
                h_conv = tf.nn.relu(
					conv2d(self.x_image, self.weights["filter"+ str(i)]) + self.biases["filter" + str(i)]
					)

                h_pool = max_pool_2x2(h_conv, self.n_words - filter + 1, 1)
				
                h_norm = tf.nn.local_response_normalization(h_pool)
                h_norm_flat = tf.reshape(
					h_norm, 
					[-1, 1 * 1 * self.n_channels]
					)
				
                convolution_outputs.append(h_norm_flat)

        conv_out = tf.concat(1, convolution_outputs)
        conv_out = tf.nn.dropout(conv_out, self.drop_keep_prob)

        hidden = tf.nn.relu(tf.matmul(conv_out, self.weights['dense1']) + self.biases['dense1'])
        hidden = tf.nn.dropout(hidden, self.drop_keep_prob)
		
        predict = tf.nn.softmax(tf.matmul(hidden, self.weights['dense2']) + self.biases['dense2'])

        return predict


    @define_scope
    def optimize(self):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            self.prediction,
            self.y
            )

        self.cost = tf.reduce_mean(cross_entropy)

        adam_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = adam_op.minimize(loss = self.cost, name = "ad_opt")
        return self.optimizer


    @define_scope
    def measure_performance(self):

        self.model_pred = tf.argmax(
            self.prediction,
            1,
            name = "predicted_value"
            )

        self.real_val = tf.argmax(self.y, 1, name = "actual_value")
        correct  = tf.equal(self.model_pred, self.real_val)
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name = "accuracy")

        #return model prediction and real_val as well to caluclate f1 score
        #outside of tensorflow
        return [self.accuracy, self.model_pred, self.real_val]


    @define_scope
    def tensorboard_summary(self):

        tf.summary.scalar("Accuracy", self.accuracy)
        tf.summary.scalar("Loss", self.cost)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        #for grad, var in grads:
        #    tf.histogram_summary(var.name + '/gradient', grad)

        self.merged_summary_op = tf.summary.merge_all()

        return self.merged_summary_op


    def info(self):

        print("Model Name:        " + self.model_name)
        print("Embedding size:    " + str(self.embed_len))
        print("filter sizes:      " + str(self.filter_sizes))
        print("n_channels:        " + str(self.n_channels))
        print("Dropout keep prob: " + str(self.drop_keep_prob))
