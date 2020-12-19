from docopt import docopt
import logging
import time
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import spearmanr


# returns correlation between measured CD and freq
def corr_CD_freq(matrix, words, targets, freq):
    freq_diff = []
    scores = []
    for (word1, word2) in targets:
        # get distance for all targets, nan if oof
        if word1 in words and word2 in words:
            vector1 = matrix[words.index(word1)]
            vector2 = matrix[words.index(word2)]
            distance = cosine_distance(vector1, vector2)
            diff = abs(freq[word1] - freq[word2])
        else:
            distance = np.nan
            diff = np.nan
        scores += [distance]
        freq_diff += [diff]
    return scores, freq_diff


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

    # length normalise matrix
    matrix_full = np.matrix(matrix_full)
    l2norm = np.linalg.norm(matrix_full, axis=1, ord=2)
    l2norm[l2norm == 0.0] = 1.0  # Convert 0 values to 1
    matrix_full /= l2norm.reshape(len(l2norm), 1)

    # compute centroid_high
    matrix_full = np.matrix(matrix_full)
    l2norm = np.linalg.norm(matrix_full, axis=1, ord=2)
    l2norm[l2norm == 0.0] = 1.0  # Convert 0 values to 1
    matrix_full /= l2norm.reshape(len(l2norm), 1)
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
        test_statistik_wordsim.py <matrixPath> <wordsim_goldPath> <freqPath> <#alpha>

    Arguments:
        <matrixPath> = path to matrix1
        <wordsim_goldPath> = path to godl data wordsim
        <freqPath> = path to freq data
    """)

    matrixPath = args['<matrixPath>']
    wordsim_goldPath = args['<wordsim_goldPath>']
    freqPath = args['<freqPath>']
    alpha = int(args['<#alpha>'])

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

    # correlation between scalar projection and freq
    # print(freq_encoding(matrix, centroid, freqs, words))

    # print labels for each column
    print('freq_diff \t scores')

    eval_matrix = matrix - alpha * centroid

    # calculate statistics
    scores, freq_diff = corr_CD_freq(eval_matrix, words, gold, logfreqs)

    # print statistics
    # result_tuple = (alpha, wordsim, log_bias, isotropy, len_centroid, cfreq_enc)
    output = ''
    for i in zip(freq_diff, scores):
        print('{:.6f}\t{:.6f}'.format(*i))

    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
