from docopt import docopt
import logging
import time
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import spearmanr


# returns correlation between measured CD and gold data
def corr_CD_wordsim(matrix, words, gold_data):
    gold = []
    scores = []
    for (word1, word2) in gold_data:
        if word1 in words and word2 in words:
            vector1 = matrix[words.index(word1)]
            vector2 = matrix[words.index(word2)]
            distance = cosine_distance(vector1, vector2)
        else:
            distance = np.nan
        scores += [distance]
        gold += [gold_data[(word1, word2)]]
    return spearmanr(scores, gold, nan_policy='omit')[0]

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
    return spearmanr(scores, freq_diff, nan_policy='omit')[0]

def word_freq_spliter(words, targets, freq):
    freq_100_200 = []
    freq_201_1000 = []
    freq_g1000 = []
    for (word1, word2) in targets:
        # get distance for all targets, nan if oof
        if word1 in words and word2 in words:
            freq_w = max(freq[word1], freq[word2])
            if 100 <= freq_w <= 200:
                freq_100_200.append((word1, word2))
            if 201 <= freq_w <= 1000:
                freq_201_1000.append((word1, word2))
            if freq_w > 1000:
                freq_g1000.append((word1, word2))

    return freq_100_200, freq_201_1000, freq_g1000


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


def isotropyANDcentroid(matrix):
    # compute centroid length
    matrix = np.matrix(matrix)
    l2norm = np.linalg.norm(matrix, axis=1, ord=2)
    l2norm[l2norm == 0.0] = 1.0  # Convert 0 values to 1
    matrix /= l2norm.reshape(len(l2norm), 1)
    # Center matrix
    avg = np.mean(matrix, axis=0)
    centroid_length = 1 - np.linalg.norm(avg).round(3)

    # compute Isotropy after l2 normalising
    V = np.matrix(matrix)
    C = np.linalg.eig(V.T * V)[1]
    Z = [v.sum() for v in np.power(np.e, C.T * V.T)]
    isotropy = np.min(Z)/np.max(Z)

    return isotropy, centroid_length


def log_freq_target_diff(matrix, words, targets, freq, centroid):
    freq_diff = []
    scalar_projection_diff = []
    for (word1, word2) in targets:
        # get distance for all targets, nan if oof
        if word1 in words and word2 in words:
            vector1 = matrix[words.index(word1)]
            vector2 = matrix[words.index(word2)]

            cent = centroid.T / np.linalg.norm(centroid.T)
            sp_1 = np.dot(vector1, cent)
            sp_2 = np.dot(vector2, cent)

            diff = abs(freq[word1] - freq[word2])
            sp_diff = (sp_1 - sp_2).tolist()

            logging.info("freq: {}, sp_diff: {}".format(diff, sp_diff[0]))
        else:
            diff = np.nan
            sp_diff = np.nan
        scalar_projection_diff += [sp_diff[0]]
        freq_diff += [diff]
    logging.info("freq: {}, sp_diff: {}".format(freq_diff, scalar_projection_diff))

    return spearmanr(freq_diff, scalar_projection_diff, nan_policy='omit')[0]


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

    # correlation between scalar projection and freq
    # print(freq_encoding(matrix, centroid, freqs, words))

    # alpha step setup
    start = -2
    end = 4
    step = 0.1
    alpha = start

    # print("centroid length: {}".format(np.linalg.norm(centroid)))

    wp1, wp2, wp3 = word_freq_spliter(words, gold, freqs)

    # print labels for each column
    print('alpha \t wordsim \t log_bias \t isotropy \t len_centroid* \t lf_sp_cor \t lb_l \t lb_m \t lb_h')

    while alpha <= end:
        logging.info("Currently at alpha=" + str(alpha))

        eval_matrix = matrix - alpha * centroid

        # calculate statistics
        wordsim = corr_CD_wordsim(eval_matrix, words, gold)
        log_bias = corr_CD_freq(eval_matrix, words, gold, logfreqs)
        lb_l = corr_CD_freq(eval_matrix, words, wp1, logfreqs)
        lb_m = corr_CD_freq(eval_matrix, words, wp2, logfreqs)
        lb_h = corr_CD_freq(eval_matrix, words, wp3, logfreqs)
        isotropy, len_centroid = isotropyANDcentroid(eval_matrix)
        lf_sp_cor = log_freq_target_diff(
            eval_matrix, words, gold, logfreqs, centroid)

        # print statistics
        result_tuple = (alpha, wordsim, log_bias,
                        isotropy, len_centroid, lf_sp_cor, lb_l, lb_m, lb_h)
        output = ''
        for entry in result_tuple:
            output += '{:.6f}\t'.format(entry)
        print(output.strip())

        # increment alpha, rounding cause python cant do maths...
        alpha = round(alpha + step, 2)

    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
