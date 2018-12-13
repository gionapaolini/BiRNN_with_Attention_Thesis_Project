import tensorflow as tf
import numpy as np
import random
from pathlib import Path
import time





class Word2Vec: 
	def __init__(self, data, target, maxLength, dropout, num_hidden=200, num_layers=2):
        
		self.treshold = 
		self.n_sampled = 
		self.n_embedding = 300
		self.learning_rate = 0.01
		self.epochs = 10
		self.window_size = 
		self.batchSize
		self.int_to_vocab
		self.data
		
		
		
		
		
		
	def get_target(words, idx, window_size=5):
		
		R = np.random.randint(1, window_size+1)
		start = idx - R if (idx - R) > 0 else 0
		stop = idx + R
		target_words = set(words[start:idx] + words[idx+1:stop+1])
		
		return list(target_words)


	def get_batches(n_sentences_in_batch, sentences, window_size=5):  

		for idx in range(0, len(sentences), n_sentences_in_batch):
			x, y = [], []
			batch = sentences[idx:idx+n_sentences_in_batch]
			for batchIndex in range(len(batch)):
				for wordIndex in range(len(batch[batchIndex])):
					batch_x = batch[batchIndex][wordIndex]
					batch_y = get_target(batch[batchIndex], wordIndex, window_size)
					y.extend(batch_y)
					x.extend([batch_x]*len(batch_y))
			yield x, y
 

def word2vec(pathSample,threshold,n_sampled,n_embedding,learning_rate,epochs,window_size,n_sentences_in_batch,int_to_vocab,int_wordsPerInteraction, train=False):
    n_vocab = len(int_to_vocab) + 1
    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs = tf.placeholder(tf.int32, [None], name='inputs')
        labels = tf.placeholder(tf.int32, [None, None], name='labels')
        
        embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1),name='embedding')
        
        embed = tf.nn.embedding_lookup(embedding, inputs)
        
        softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev=0.1), name='softmax_w')
        softmax_b = tf.Variable(tf.zeros(n_vocab),name='softmax_b')

        # Calculate the loss using negative sampling
        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, 
                                          labels, embed,
                                          n_sampled, n_vocab)

        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        
        
        valid_size = 16 # Random set of words to evaluate similarity on.
        valid_window = 100
        # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent 
        valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
        valid_examples = np.append(valid_examples, 
                                   random.sample(range(1000,1000+valid_window), valid_size//2))

        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embedding = embedding / norm
        valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
        similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
        
        saver = tf.train.Saver()
        
    my_file = Path("checkpoints/{}_{}_{}_{}_{}_{}_{}_{}.ckpt.meta".format(pathSample,threshold,n_sampled,n_embedding,learning_rate,epochs,window_size,n_sentences_in_batch))
    with tf.Session(graph=train_graph) as sess:
        if my_file.is_file():
            new_saver = tf.train.import_meta_graph("checkpoints/{}_{}_{}_{}_{}_{}_{}_{}.ckpt.meta".format(pathSample,threshold,n_sampled,n_embedding,learning_rate,epochs,window_size,n_sentences_in_batch))
            new_saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            #embeddings = sess.graph.get_tensor_by_name("embedding:0")
            #sess.run(embeddings)
            #embed_mat = sess.run(normalized_embedding)
        
        if(train): 
            print("Start training")
            iteration = 1
            loss = 0
            #sess.run(tf.global_variables_initializer())

            for e in range(1, 10000):
                sentences = get_batches(n_sentences_in_batch ,int_wordsPerInteraction, window_size)
                start = time.time()
                for x, y in sentences:

                    feed = {inputs: x,
                            labels: np.array(y)[:, None]}
                    train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                    loss += train_loss

                    
                    if iteration % 100 == 0: 
                        end = time.time()
                        print("Epoch {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss/100),
                              "{:.4f} sec/batch".format((end-start)/100))
                        loss = 0
                        start = time.time()

                    if iteration % 1000 == 0:
                        # note that this is expensive (~20% slowdown if computed every 500 steps)
                        sim = similarity.eval()
                        for i in range(valid_size):
                            valid_word = int_to_vocab[valid_examples[i]]
                            top_k = 8 # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k+1]
                            log = 'Nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = int_to_vocab[nearest[k]]
                                log = '%s %s,' % (log, close_word)
                            print(log)

                    iteration += 1
                
                save_path = saver.save(sess, "checkpoints/word2vec{}.ckpt".format(e))
            
            embed_mat = sess.run(normalized_embedding)
        
    return embed_mat
   