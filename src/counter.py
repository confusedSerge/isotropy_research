from docopt import docopt
import logging
import time
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import spearmanr


def word_freq_counter(words, targets, freq):
    # TODO: counter
    freq_c = [0, 0, 0, 0]
    for (word1, word2) in targets:
        freq_c[_fadd(freq[word1], freq_c)] += 1
        freq_c[_fadd(freq[word2], freq_c)] += 1
    return freq_c


def _fadd(fw, fc):
    if fw < 100:
        return 0
    elif 100 <= fw <= 200:
        return 1
    elif 201 <= fw <= 1000:
        return 2
    return 4

def loadMatrix(path):
    matrix_full = []
    word_index = []

    # read matrix
    with open(path) as f_in:
        f_in.readline()  # ignore header
        for line in f_in.readlines():
            split = line.split(' ')
            word = split[0]
            vector = np.array(split[1:]).astype(np.float)
            matrix_full += [vector]
            word_index += [word]

    # normalize and compute centroid_high
    matrix_full = np.matrix(matrix_full)
    centroid = np.mean(matrix_full, axis=0)

    return matrix_full, word_index, centroid


def main():
    """
    prints various statistiks for input matrix (tab saperated):
    name,
    wordsim, frequency, isotropy, len_centroid
    """

    # Get the arguments
    args = docopt("""various statistiks for input matrix
    Usage:
        test_statistik_wordsim.py <matrixPath> <wordsim_goldPath> <freqPath>

    Arguments:
        <matrixPath> = path to matrix1
        <wordsim_goldPath> = path to godl data wordsim
        <freqPath> = path to freq data
    """)

    matrixPath = args['<matrixPath>']
    wordsim_goldPath = args['<wordsim_goldPath>']
    freqPath = args['<freqPath>']

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # load wordsim gold score file
    with open(wordsim_goldPath, 'r', encoding='utf-8') as f_in:
        # dict (word1,word2):score
        gold = {tuple(line.strip().split('\t')[0:2]): float(
            line.strip().split('\t')[-1]) for line in f_in}

    # load freq file
    with open(freqPath, 'r', encoding='utf-8') as f_in:
        # dict word [tab] freq
        freqs = {line.strip().split('\t')[0]: int(line.strip().split(
            '\t')[1]) for line in f_in}
        # dict log freq
        logfreqs = {word: np.log10(freqs[word]) for word in freqs}

    # laod matrices
    logging.info("Read Matrix ...")
    matrix, words, centroid = loadMatrix(matrixPath)

    # somehow there are words in words which are not in freqs,
    # those will be added with nan values
    for word in words:
        if word not in freqs:
            freqs[word] = np.nan

    freq_c = word_freq_counter(words, gold, freqs)

    print("<100 \t 100 - 200 \t 201 - 1000 \t >1000")
    print(freq_c)

    logging.info("--- %s seconds ---" % (time.time() - start_time))


# if __name__ == '__main__':
#     main()

bla = [0, 0]
bla[0] += 1
print(bla)