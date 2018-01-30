import numpy as np
import tensorflow as tf

class ConvAutoEncoder:
	def __init__(self,model_name,keep_prob,learning_rate,dim,look_around,feature_dim=256):
		self.dim = dim
		self.feature_dim = feature_dim
		self.look_around = look_around

		self.keep_prob = keep_prob
		self.model_name = model_name
		self.learning_rate = learning_rate

		self.stride = 2
		self.ksize = 3

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



	def _maxpool1D(self,inp,ksize,strides,padding,name=None):
		return tf.squeeze(tf.nn.max_pool(tf.expand_dims(inp,axis=1),(1,1,ksize,1),(1,1,strides,1),padding,name=name),axis=1)

	# def _unmaxpool1D(self,inp,binary_mask,is_training,strides,padding,name=None):
	# 	repeated_tensor = self._repeat(inp,[1,strides,1])
	# 	if(is_training):
	# 		return repeated_tensor*binary_mask
	# 	else:
	# 		return repeated_tensor/2
	def _unmaxpool1D(self,inp,strides,padding,name=None):
		repeated_tensor = self._repeat(inp,[1,strides,1])
		return repeated_tensor/2

	def _repeat(self,tensor,repeats):
	    with tf.variable_scope("repeat"):
	        expanded_tensor = tf.expand_dims(tensor, -1)
	        multiples = [1] + repeats
	        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
	        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
	    return repeated_tensor
	
	
	def contruct_network(self):

		self.encoder_W = tf.Variable(tf.truncated_normal([self.look_around,self.dim[1],self.feature_dim]))
		self.encoder_b = tf.Variable(tf.zeros([self.feature_dim]))

		self.wx = tf.nn.conv1d(self.x,self.encoder_W,stride=1,padding='SAME')
		self.wx_b = tf.nn.bias_add(self.wx,self.encoder_b)
		self.mp_wx_b = self._maxpool1D(self.wx_b,self.ksize,self.stride,'SAME')
		self.dropout_mp_wx_b = tf.nn.dropout(self.mp_wx_b,keep_prob=self.dropout)
		self.feature = tf.nn.relu(self.dropout_mp_wx_b)

		self.wx_b_ = self._unmaxpool1D(self.feature,self.stride,padding='SAME',name=None)
		self.wx_ = tf.nn.bias_add(self.wx_b_,-self.encoder_b)
		self.dropout_wx_ = tf.nn.dropout(self.wx_,keep_prob=self.dropout)
		self.output = tf.nn.conv1d(self.dropout_wx_,tf.transpose(tf.reverse(self.encoder_W,axis=[0]),perm=(0,2,1)),stride=1,padding="SAME")

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



