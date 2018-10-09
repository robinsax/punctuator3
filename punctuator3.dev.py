import sys
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

class Config:

	def __init__(self, data):
		self.data = data

	def __getattr__(self, attr):
		return self.data[attr]
	
	def __setattr__(self, attr, value):
		if attr == 'data':
			super().__setattr__(attr, value)
		self.data[attr] = value

config = Config({
	'minibatch_size': 10,
	'n_hidden': 20,
	'train_epochs': 2,
	'model_dir': './test-model/'
})

def s_print(*ts):
	for t in ts:
		if isinstance(t, str):
			print(t)
			continue
		print(t.get_shape().as_list())

class GRU:

	def __init__(self, name, in_dim, out_dim):
		self.in_dim, self.out_dim = in_dim, out_dim

		with tf.variable_scope(name):
			#	Initial activation.
			self.h_0 = tf.zeros(
				(config.minibatch_size, out_dim), 
				name='h_0'
			)
			#	Composite update/reset gate weights.
			self.W_rz = tf.get_variable(
				'W_rz', (in_dim, out_dim*2),
				tf.float32, tf.glorot_uniform_initializer()
			)
			self.U_rz = tf.get_variable(
				'U_rz', (out_dim, out_dim*2), 
				tf.float32, tf.glorot_uniform_initializer()
			)
			#	Composite update/reset gate bias.
			self.b_rz = tf.get_variable(
				'b_rz', (1, out_dim*2),
				tf.float32, tf.zeros_initializer()
			)
			#	Activation weights and biases.
			self.W_h = tf.get_variable(
				'W_h', (in_dim, out_dim),
				tf.float32, tf.glorot_uniform_initializer()
			)
			self.U_h = tf.get_variable(
				'U_h', (out_dim, out_dim),
				tf.float32, tf.glorot_uniform_initializer()
			)
			self.b_h = tf.get_variable(
				'b_h', (1, out_dim),
				tf.float32, tf.zeros_initializer()
			)

		self.variables = (
			self.W_rz, self.U_rz, self.b_rz, self.W_h, self.U_h, self.b_h
		)

	def step(self, h_tm1, x_t):
		#	Compute composite gate activation...
		rz_t = tf.sigmoid(
			tf.matmul(x_t, self.W_rz) + 
			tf.matmul(h_tm1, self.U_rz) + self.b_rz
		)
		#	...and unpack.
		r_t = tf.slice(rz_t, [0, 0], [1, self.out_dim])
		z_t = tf.slice(rz_t, [0, 1], [1, self.out_dim])

		#	Compute activation.
		h_activation = tf.tanh(
			tf.matmul(x_t, self.W_h) + 
			tf.matmul(h_tm1 * r_t, self.U_h) + self.b_h
		)

		#	Apply update gateing.
		h_t = z_t*h_tm1 + (1.0 - z_t)*h_activation
		return h_t

#	Context attention mechanism
class CAM:

	def __init__(self):
		n_attention = config.n_hidden*2

		with tf.variable_scope('context_attention'):
			#	Output model previous activation weights.
			self.Wa_h = tf.get_variable(
				'Wa_h', (config.n_hidden, n_attention),
				tf.float32, tf.glorot_uniform_initializer()
			)
			#	Context attention weights.
			self.Wa_c = tf.get_variable(
				'Wa_c', (n_attention, n_attention),
				tf.float32, tf.glorot_uniform_initializer()
			)
			#	Attention output weights.
			self.Wa_y = tf.get_variable(
				'Wa_y', (n_attention, 1),
				tf.float32, tf.glorot_uniform_initializer()
			)
			#	Attention bias.
			self.ba = tf.get_variable(
				'ba', (1, n_attention),
				tf.float32, tf.zeros_initializer()
			)

		self.variables = (
			self.Wa_h, self.Wa_c, self.Wa_y, self.ba
		)

	def project_context(self, context):
		return tf.tensordot(context, self.Wa_c, [[2], [0]]) + self.ba

	def weight_context(self, context, proj_context, h_tm1):
		#	Compute activation.
		ha_t = tf.tanh(proj_context + tf.matmul(h_tm1, self.Wa_h))
		#	Compute alphas.
		alphas = tf.exp(tf.tensordot(ha_t, self.Wa_y, [[2], [0]]))
		alphas_shape = tf.shape(alphas)
		alphas = tf.slice(alphas, [0, 0, 0], [alphas_shape[0], alphas_shape[1], 1])
		#	Normalize.
		alphas = alphas / tf.reduce_sum(alphas, keepdims=True)
		weighted_context = tf.reduce_sum(context * alphas, axis=0)

		return weighted_context

class LateFuser:

	def __init__(self):
		n_attention = config.n_hidden*2

		with tf.variable_scope('late_fusion'):
			#	Activation fusion weights.
			self.Wf_h = tf.get_variable(
				'Wf_h', (config.n_hidden, config.n_hidden),
				tf.float32, tf.glorot_uniform_initializer()
			)
			#	Context fusion weights.
			self.Wf_c = tf.get_variable(
				'Wf_c', (n_attention, config.n_hidden),
				tf.float32, tf.glorot_uniform_initializer()
			)
			#	Fusion output weights and bias.
			self.Wf_y = tf.get_variable(
				'Wf_y', (config.n_hidden, config.n_hidden),
				tf.float32, tf.glorot_uniform_initializer()
			)
			self.bf = tf.get_variable(
				'bf', (1, config.n_hidden),
				tf.float32, tf.zeros_initializer()
			)

		self.variables = (
			self.Wf_h, self.Wf_c, self.Wf_y, self.bf
		)

	def late_fuse(self, h_t, weighted_context):
		#	Perform fusion.
		late_fused_context = tf.matmul(weighted_context, self.Wf_c)
		fusion_weights = tf.sigmoid(
			tf.matmul(late_fused_context, self.Wf_y) + 
			tf.matmul(h_t, self.Wf_h) + self.bf
		)

		#	Compute activation.
		hf_t = late_fused_context*fusion_weights*h_t
		return hf_t

class PuncModel:

	def __init__(self, word_vocab, punc_vocab):
		self.word_vocab, self.punc_vocab = word_vocab, punc_vocab

		#	Word embeddings.
		self.W_e = tf.get_variable(
			'W_e', (len(word_vocab), config.n_hidden),
			tf.float32, tf.glorot_uniform_initializer()
		)

		#	Bi-directional units.
		self.gru_f = GRU('gru_f', config.n_hidden, config.n_hidden)
		self.gru_b = GRU('gru_b', config.n_hidden, config.n_hidden)
	
		#	Context attention model.
		self.cam = CAM()
		#	Late fusion mechanism.
		self.late_fuser = LateFuser()

		#	Output unit, weights and bias.
		self.gru_y = GRU('gru_y', config.n_hidden*2, config.n_hidden)
		self.W_y = tf.get_variable(
			'W_y', (config.n_hidden, len(punc_vocab)),
			tf.float32, tf.glorot_uniform_initializer()
		)
		self.b_y = tf.get_variable(
			'b_y', (1, len(punc_vocab)),
			tf.float32, tf.zeros_initializer()
		)
		
		#	Create context...
		self.context = self._create_context()
		#	...and project.
		self.proj_context = self.cam.project_context(self.context)

		#	Create graph.
		#self._scan_step(self.gru_y.h_0, self.context[0])
		y_0 = tf.zeros([config.minibatch_size, len(punc_vocab)], name='y_0')
		meta_init = (self.gru_y.h_0, y_0)
		self.y = tf.scan(self._scan_step, self.context, meta_init)[1]
		tf.identity(self.y) # For graph-view purposes.

		self.variables = (
			self.W_e, *self.gru_f.variables, *self.gru_b.variables,
			*self.cam.variables, *self.late_fuser.variables, *self.gru_y.variables,
			self.W_y, self.b_y
		)

	def _create_context(self):
		#	Gather embedded sequences.
		x_pl = tf.placeholder(tf.int32, [None, config.minibatch_size], 'x')
		x_emb_seq = tf.reshape(
			tf.gather(self.W_e, tf.reshape(x_pl, [-1])),
			(tf.shape(x_pl)[0], config.minibatch_size, config.n_hidden)
		)
		rev_x_emb_seq = tf.reverse(x_emb_seq, [1])

		#	Get forward and reverse scans.
		hf = tf.scan(self.gru_f.step, x_emb_seq, self.gru_f.h_0)
		hb = tf.scan(self.gru_b.step, rev_x_emb_seq, self.gru_b.h_0)
		return tf.concat([
			hf, tf.reverse(hb, [1])
		], 2)

	def _scan_step(self, meta, x_t):
		h_tm1, y_tm1 = meta
		#	Get attention-weighted context.
		weighted_context = self.cam.weight_context(self.context, self.proj_context, h_tm1)
		#	Compute activation.
		h_t = self.gru_y.step(h_tm1, x_t)
		#	Perform late fusion.
		hf_t = self.late_fuser.late_fuse(h_t, weighted_context)

		#	Get output.
		y_t = tf.nn.softmax(tf.matmul(hf_t, self.W_y) + self.b_y)
		y_t.set_shape([config.minibatch_size, len(self.punc_vocab)]) #XXX ???
		return h_t, y_t

	def run(self, x, sess):
		return sess.run(self.y, {'x:0': x})

def punctuate(word_vocab, punc_vocab, x):
	#	Execute model.
	with tf.Session() as sess:
		model = PuncModel(word_vocab, punc_vocab)

		sess.run(tf.global_variables_initializer())
		y = model.run(x, sess)
	
	#	Transform.
	y = list(np.argmax(y_t[0]) for y_t in y)
	x = np.array(x).flatten().tolist()
	
	#	Project into vocabulary.
	out_tokens = list()
	for x_t, y_t in zip(x, y):
		out_tokens.extend((
			word_vocab[x_t],
			punc_vocab[y_t]
		))
	
	return ''.join(out_tokens)

def train_model(word_vocab, punc_vocab, datasets):
	with tf.Session() as sess:
		model = PuncModel(word_vocab, punc_vocab)
		with sess.graph.as_default():
			saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())

		for i in range(config.train_epochs):
			for j, batch in enumerate(datasets):
				x, y = batch

				loss_op = tf.losses.sparse_softmax_cross_entropy(y, model.y)
				optimizer = tf.train.AdamOptimizer()
				optimizer_op = optimizer.minimize(loss_op, var_list=model.variables)


				sess.run(tf.variables_initializer(optimizer.variables()))
				print(
					'%s/%s'%(i, config.train_epochs), 
					'%s/%s'%(j, len(datasets)),
					sess.run([
						loss_op, 
						optimizer_op
					], feed_dict={'x:0': x})[0]
				)

		saver.save(sess, config.model_dir)
				
if __name__ == '__main__':
	word_vocab = {0: 'a', 1: 'b', 2: 'c'}
	punc_vocab = {0: ' ', 1: '-', 2: '+'}
	
	mode = sys.argv[1]
	if mode == 'train':
		import fake_corpus
		datasets = fake_corpus.create_fake_corpus(word_vocab, punc_vocab, 50, 3, config.minibatch_size)

		train_model(word_vocab, punc_vocab, datasets)
	elif mode == 'run':
		config.minibatch_size = 1

		#	Execute model.
		with tf.Session() as sess:
			model = PuncModel(word_vocab, punc_vocab)
			with sess.graph.as_default():
				saver = tf.train.Saver()

			sess.run(tf.global_variables_initializer())
			saver.restore(sess, config.model_dir)
			
			while True:
				x_sym = input('Symbols: ')
				x = []
				for x_t in x_sym:
					for k, v in word_vocab.items():
						if v == x_t:
							x.append([k])

				y = model.run(x, sess)
				
				#	Transform.
				y = list(np.argmax(y_t[0]) for y_t in y)
				x = np.array(x).flatten().tolist()
				
				#	Project into vocabulary.
				out_tokens = list()
				for x_t, y_t in zip(x, y):
					out_tokens.extend((
						word_vocab[x_t],
						punc_vocab[y_t]
					))
				
				print(''.join(out_tokens))
	else:
		raise ValueError(mode)