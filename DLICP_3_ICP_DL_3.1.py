
import collections
import math
import os
import errno
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
from collections import Counter
import tensorflow as tf

data_dir = "data"
data_url = 'http://a.net/dc/text8.zip'


def fetch_words_data(url=data_url, words_data=data_dir):

    os.makedirs(words_data, exist_ok=True)
    zip_path = os.path.join(words_data, "words.zip")


    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])

    return data.decode("ascii").split()


words = fetch_words_data()

# Total words
print("Total length of words is: ", len(words))

for w in words[9000:9040]:
    print(w, end=' ')



def create_counts(vocab_size=50000):
    vocab = [] + Counter(words).most_common(vocab_size)

    vocab = np.array([word for word, _ in vocab])

    dictionary = {word: code for code, word in enumerate(vocab)}
    data = np.array([dictionary.get(word, 0) for word in words])
    return data, vocab


vocab_size = 50000

data, vocabulary = create_counts(vocab_size=vocab_size)



def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
        buffer[:] = data[:span]
        data_index = span
    else:
        buffer.append(data[data_index])
        data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


data_index = 0
batch, labels = generate_batch(8, 2, 1)

batch_size = 128

embedding_size = 150                                                            # changed for #3 to 50

skip_window = 1

num_skips = 2


valid_size = 16                                                                 # CHANGED to to 50 QUESTION FOR #2

valid_window = 200
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

num_sampled = 64

learning_rate = 0.01                                                            # Changed to 0.09 for Q1

vocabulary_size = 50000


tf.reset_default_graph()

train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)

embed = tf.nn.embedding_lookup(embeddings, train_inputs)


nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                   num_sampled, vocabulary_size))


optimizer = tf.train.AdamOptimizer(learning_rate=1.0)
trainer = optimizer.minimize(loss)


norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

init = tf.global_variables_initializer()


num_steps = 5000                                                             # Q4 initially changed in step one due to
                                                                             # length of runtime, changed again to 2000
                                                                             # steps to show the difference

                                                                             # Q5 FileWriter(source, sess.graph)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('/Users/marcpepperman/Desktop/DeepLearning_ICP_CS490/DLICP_3', sess.graph)
    sess.run(init)
    average_loss = 0
    for step in range(num_steps):

        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        empty, loss_val = sess.run([trainer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 1000 == 0:
            if step > 0:
                average_loss /= 1000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

    final_embeddings = normalized_embeddings.eval()


import matplotlib.pyplot as plt


def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')


from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 2000
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])

labels = [vocabulary[i] for i in range(plot_only)]

plot_with_labels(low_dim_embs, labels)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
