"""Sample code SVD."""
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os


def main():
    la = np.linalg

    words = ['i', 'like', 'enjoy', 'deep', 'learning', 'NLP', 'flying', '.']
    x = np.array([
        [0, 2, 1, 0, 0, 0, 0, 0],
        [2, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
    ])

    u, s, vh = la.svd(x, full_matrices=False)

    for i in range(len(words)):
        plt.text(u[i, 0], u[i, 1], words[i])

    plt.savefig('svd')
    subprocess.call(['catimg', '-f', 'svd.png'])
    os.remove('svd.png')

if __name__ == '__main__':
    main()



