import numpy as np
import tensorflow as tf

class ConvAutoEncoder:
	def __init__(self,model_name,keep_prob,learning_rate,dim):
		self.dim = dim

		self.keep_prob = keep_prob
		self.model_name = model_name
		self.learning_rate = learning_rate

		self.declare_placeholder()
		self.contruct_network()
		self.define_loss()
		self.define_optimizer()
		self.define_saver()
		self.create_session()
		self.declare_init()
		self.run_init()

	def declare_placeholder(self):
		self.x = tf.placeholder(tf.float32,shape=(None,self.dim[0],self.dim[1]))
		self.dropout = tf.placeholder(tf.float32)


	def contruct_network(self):

		self.encoder_W = tf.Variable(tf.truncated_normal([self.dim[0]*self.dim[1],int(self.dim[0]*self.dim[1]/2)]))
		self.encoder_b = tf.Variable(tf.truncated_normal([int(self.dim[0]*self.dim[1]/2)]))

		# self.decoder_1W = tf.Variable(tf.truncated_normal([int(self.dim[0]*self.dim[1]/4),int(self.dim[0]*self.dim[1]/2)]))
		# self.decoder_1b = tf.Variable(tf.truncated_normal([int(self.dim[0]*self.dim[1]/2)]))
		# self.decoder_2W = tf.Variable(tf.truncated_normal([int(self.dim[0]*self.dim[1]/2),self.dim[0]*self.dim[1]]))
		# self.decoder_2b = tf.Variable(tf.truncated_normal([self.dim[0]*self.dim[1]]))

		self.wxb = tf.nn.conv1d()

		self.enc1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(tf.reshape(self.x,[-1,self.dim[0]*self.dim[1]]),\
			self.encoder_1W),self.encoder_1b))
		self.enc1 = tf.nn.dropout(self.enc1,keep_prob = self.dropout)
		self.feat = tf.nn.tanh(tf.nn.bias_add(tf.matmul(self.enc1,self.encoder_2W),self.encoder_2b))
		self.feat = tf.nn.dropout(self.feat,keep_prob = self.dropout)

		self.dec1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(self.feat,self.decoder_1W),self.decoder_1b))
		self.dec1 = tf.nn.dropout(self.dec1,keep_prob = self.dropout)
		self.output = tf.reshape(tf.nn.bias_add(tf.matmul(self.dec1,self.decoder_2W),self.decoder_2b),[-1,self.dim[0],self.dim[1]])
		self.output = tf.nn.dropout(self.output,keep_prob = self.dropout)

	def define_loss(self):
		self.loss = tf.reduce_mean(tf.square(self.output-self.x))

	def define_optimizer(self):
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

	def declare_init(self):
		self.init = tf.global_variables_initializer()

	def run_init(self):
		self.session.run(self.init)

	def create_session(self):
		self.session = tf.Session()

	def define_saver(self):
		self.saver = tf.train.Saver()

	def save_model(self):
		self.saver.save(self.session,self.model_name)

	def restore_model(self):
		self.saver.restore(self.session,self.model_name)
		
	def run_training(self,input):
		self.session.run(self.optimizer,feed_dict={self.x:input,self.dropout:self.keep_prob})
	def get_loss(self,input):
		return self.session.run(self.loss,feed_dict={self.x:input,self.dropout:1})



