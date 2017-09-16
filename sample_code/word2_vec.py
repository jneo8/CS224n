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

################################
# step 1: Download the data.
###############################

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(gettempdir(), filename)
    print(local_filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(
                url + filename, local_filename
        )
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print(f'Found and verified {filename}')  # noqa
    else:
        print(statinfo.st_size)
        raise exception(
            f'Failed to verify {local_filename}.'
            ' Can you get to it with a browser? '
        )
    return local_filename

filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))



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








