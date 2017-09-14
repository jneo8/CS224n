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












