"""Word2vec use tensorflow example."""

import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))

# logger
try:
    from logger import logconf
    logger = logconf.Logger(__name__)
except:
    raise ImportError('Please install logger')

logger.info('Start')


# Read the data into a list of strings.
def read_data():
    """Extract the first enclosed in a zip file as a list of words."""
    file_path = os.path.join(dir_path, 'source', 'ch_line.txt')
    data = []
    with open(file_path, 'r') as lines:
        for line in lines:
            line = line.split(' ')
            data += line

    return data

vocabulary = read_data()
print('Data size', len(vocabulary))
logger.debug(vocabulary[:200])


################################
# step 2: Build the dictionary and rare words with UNK token.
###############################

vocabulary_size = 50000


def build_dataset(words, n_words):
    """Process raw input into a dateset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words -1))
    dictionary = dict()
    for word, _ in count:
        logger.debug(f'{word} {_}')
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict((zip(dictionary.values(), dictionary.keys())))
    return data, count, dictionary, reversed_dictionary

################################
# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
################################

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)

logger.debug(f'{data[:10]}')

del vocabulary  # Hint to reduce memory.
print("Most common words (+UNK) ", count[:5])
print('Sample data ', data[:10], [reverse_dictionary[i] for i in data[:10]])


data_index = 0

################################
# Step 3: Function to generate a training batch for the skip-gram model.
################################

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [skip_window target skip_window]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words)
        for j in range(num_skips):
            batch[i * num_skips + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch

    data_index = (data_index + len(data) - span % len(data))
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(
        batch[i],
        reverse_dictionary[batch[i]],
        '->',
        labels[i, 0],
        reverse_dictionary[labels[i, 0]]
    )

################################
# Step 4: build and train a skip-gram model.
################################

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to conider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors.
# Here we limit the validatino samples to the words that have a low numeric ID,
# which by construction are also the most frequnt.
# These 3 variables are used to only for displaying model accuracy,
# they don't affect calculation.
valid_size = 16       # Random set of words to evaluate similarity on.
valid_window = 100    # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()


with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dateset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embedding for input.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        )
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size))
        )

        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Computer the average NCE loss for batch
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the eaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=num_sampled,
                        num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rare of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minbatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dateset)
    similarity = tf.matmul(
    valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()


################################
# Step 5: Begin training
################################

num_steps = 1000

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op
        # including it in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 100 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of
            # the loss over the last 2000 batches.
            logger.debug(f'Average loss at step {step}: {average_loss}')
            average_loss = 0

        if step % 1000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = f'Nearest to {valid_word} :'
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = f'{log_str} {close_word}'
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


################################
# Step 6: Visualize the embeddings.
################################

raise ValueError
# function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom'
        )

        plt.savefig(filename)
        subprocess.call(['catimg', '-f', filename])

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import subprocess
    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(
        low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)




