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
print("Most common words (+UNK), count[:5]")
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])











