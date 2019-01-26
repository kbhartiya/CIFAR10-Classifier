import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import os
import warnings; warnings.filterwarnings("ignore")
from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model',metavar='model_name',required=True,help="Enter Softmax- for CNN with softmax or SVM- for CNN with Multiclass SVM classifier")
parser.add_argument('-e','--epochs',metavar='Epoch_num',required=True,help="Enter the Number of Epochs")
parser.add_argument('-b','--batch_size',metavar='batch_size',required=True,help="Enter the Batch Size")

args = parser.parse_args()

print(args)

checkpoint_dir = './Checkpoints'
if not os.path.exists(checkpoint_dir):
	os.mkdir(checkpoint_dir)


            
x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y_true = tf.placeholder(tf.float32,shape=[None,10])
hold_prob = tf.placeholder(tf.float32)

cifar_dir = './CIFAR_Batches/'

def unpickle(file):
	with open(file,'rb') as f:
		cifar_dict = pickle.load(f,encoding='bytes')
	return cifar_dict
dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
all_data = [0,1,2,3,4,5,6]



for i,direc in zip(all_data,dirs):
	all_data[i] = unpickle(cifar_dir+direc)

batches_meta = all_data[0]
data_batch_1 = all_data[1]
data_batch_2 = all_data[2]
data_batch_3 = all_data[3]
data_batch_4 = all_data[4]
data_batch_5 = all_data[5]
test_batch = all_data[6]
	
print("Data Loaded\n")

class Cifar_model():
    def __init__(self):
        self.i = 0
        self.training_images = None
        self.training_labels = None
        self.data_batches = [data_batch_1,data_batch_2,data_batch_3,data_batch_4,data_batch_5]
        self.test_images = None
        self.test_labels = None
        self.test_batch = [test_batch]
       
    def set_up_images(self):
        print("Setting up training images and labels")
        self.training_images = np.vstack(d[b'data'] for d in self.data_batches)
        self.training_labels = one_hot_encode(np.hstack(d[b'labels'] for d in self.data_batches),10)
        training_len = len(self.training_images)
        self.training_images = self.training_images.reshape(training_len,3,32,32).transpose(0,2,3,1)/255
        
        print("Setting up testing images and labels")
        self.test_images = np.vstack(d[b'data'] for d in self.test_batch)
        self.test_labels = one_hot_encode(np.hstack(d[b'labels'] for d in self.test_batch),10)
        testing_len = len(self.test_images)
        self.test_images = self.test_images.reshape(testing_len,3,32,32).transpose(0,2,3,1)/255
        
    def next_batch(self,batch_size,purpose):
        x = self.training_images[self.i:self.i+batch_size].reshape(batch_size,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        x_test = self.test_images[self.i:self.i+batch_size].reshape(batch_size,32,32,3)
        y_test = self.test_labels[self.i:self.i+batch_size]
         
        self.i = (self.i + batch_size) % len(self.training_images)
        if purpose=="train":
        	return x,y
        else:
        	return x_test,y_test	

def one_hot_encode(vec,vals):
    n = len(vec)
    out = np.zeros((n,vals))
    out[range(n),vec] = 1
    return out

ch = Cifar_model()
ch.set_up_images()		

def convolutional_layer(input_x, shape):
	init_random_dist = tf.truncated_normal(shape, stddev=0.1)
	W = tf.Variable(init_random_dist)
	init_bias_vals = tf.constant(0.1, shape=[shape[3]])
	b = tf.Variable(init_bias_vals)

	return tf.nn.relu(tf.nn.conv2d(input_x, W,strides=[1, 1, 1, 1], padding='SAME') + b)
	
def max_pool_2by2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')	
	
def normal_full_layer(input_layer, size):
	input_size = int(input_layer.get_shape()[1])
	init_random_dist = tf.truncated_normal([input_size, size], stddev=0.1)
	W = tf.Variable(init_random_dist)
	init_bias_vals = tf.constant(0.1, shape=[size])
	b = tf.Variable(init_bias_vals)
	
	return tf.matmul(input_layer, W) + b
def _model(img_input):
    
	convo_1 = convolutional_layer(img_input,shape=[4,4,3,32])	
	convo_1_pooling = max_pool_2by2(convo_1)
	convo_2 = convolutional_layer(convo_1_pooling,shape=[4,4,32,64])
	convo_2_pooling = max_pool_2by2(convo_2)
	convo_2_flat = tf.reshape(convo_2_pooling,[-1,8*8*64])
	full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))
	full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)	
	y_pred = normal_full_layer(full_one_dropout,10)
	
	return y_pred
	
y_pred = _model(x)
epochs = int(args.epochs)
batch_size = int(args.batch_size)

saver = tf.train.Saver()

if args.model=="Softmax":
	print("--------Loading Softmax CLassifier--------------------------------------------")
	print("-----------Calculating Loss-----------------------------------------------")	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	train = optimizer.minimize(cross_entropy)
	init = tf.global_variables_initializer()
	print()
	with tf.Session() as sess:
		sess.run(init)
		print("-----------Running Epochs---------------------------------------------->")	
		num_batches = ch.training_images.shape[0] // batch_size
		for epoch in range(epochs):
			print("----------->Feeding Batches------------------------------------------>")
			for i in range(num_batches):
				batch = ch.next_batch(batch_size,"train")
				sess.run(train,feed_dict={x:batch[0],y_true:batch[1],hold_prob:0.2})
				if(i%100==0):
					print("Accuracy on step:{}".format(i))
					matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
					acc = tf.reduce_mean(tf.cast(matches,tf.float32))
					test_batch = ch.next_batch(10,"test")
					sess.run(acc,feed_dict={x:ch.test_images,y_true:ch.test_labels,hold_prob:1.0})
			if epoch%5==0:
				saver.save(sess,checkpoint_dir+"/softmax/model_epoch_"+str(epoch)+".ckpt")		
						
if args.model=="SVM":
	print("----------Loading SVM Model----------------------------------------")
	c = 0.5
	## SVM Loss
	regularization_loss = 0.5*(tf.reduce_sum(tf.square(y_pred)))
	hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([batch_size]),1-y_true*y_pred))
	svm_loss = regularization_loss + c*hingeloss
	print("---------Calculating The Loss------------------------------------")
	
	##
	train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		num_batches = ch.training_images.shape[0] // batch_size
		
		print("____________Running Epochs------------------------------------------------->")
			
		for epoch in range(epochs):
			print("----------->Feeding Batches------------------------>")
			for i in range(num_batches):
				batch = ch.next_batch(batch_size,"train")
				sess.run(train,feed_dict={x:batch[0],y_true:batch[1],hold_prob:0.2})
				if(i%100==0):
					print("============STATS=========================================================")
					print("Accuracy on step:{}".format(i))
					matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
					acc = tf.reduce_mean(tf.cast(matches,tf.float32))
					test_batch = ch.next_batch(10,"test")
					sess.run(acc,feed_dict={x:ch.test_images,y_true:ch.test_labels,hold_prob:1.0})
				if epoch%5==0:
					saver.save(sess,checkpoint_dir+"/svm/model_epoch_"+str(epoch)+".ckpt")
	
	
	
		
						


