from random import randint

def create_fake_corpus(w_v, p_v, num_items, num_batchs, batch_size):
	dataset = []
	for i in range(num_items):
		w_seq = []
		p_seq = []
		for j in range(num_batchs):
			batch_w = []
			batch_p = []
			for l in range(batch_size):
				k = randint(0, len(w_v) - 1)
				batch_w.append(k)
				batch_p.append((k + 1) % len(p_v))
			w_seq.append(batch_w)
			p_seq.append(batch_p)
		dataset.append((w_seq, p_seq))
	return dataset
