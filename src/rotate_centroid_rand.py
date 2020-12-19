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


def freq_encoding(matrix, vector, freqs, words):
    """
    returns the correlation between scalar projection of vector and frequency
    formula for scalar projection: (M * v) / (||v||)
    """
    # set vector length to 1.
    vector = vector / np.linalg.norm(vector, axis=1, ord=2)
    scalar_projection_values = np.array((matrix * vector.T)).squeeze().tolist()
    freq_values = [freqs[word] for word in words]
    return spearmanr(scalar_projection_values, freq_values,nan_policy='omit')[0]

def mean_scalar_proj(matrix, centroid):
    """
    Returns the mean scalar projection of the matrix onto the centroid
    """
    return np.mean(np.dot(matrix, centroid.T / np.linalg.norm(centroid.T)))

def rotate_centroid_rand(centroid):
    return np.linalg.norm(centroid) * np.random.uniform(-1, 1, len(centroid))

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
    rotations = int(args['<#randrot>'])
    multiplier = int(args['<#multiplier>'])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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

    # print labels for each column
    print('alpha \t wordsim \t freq_bias \t log_bias \t isotropy \t len_centroid* \t meanval \t CS_rand')

    while alpha <= end:
        logging.info("Currently at alpha=" + str(alpha))

        rand_vec = rotate_centroid_rand(centroid)
        # logging.info("Currently at alpha=" + str(alpha))

        # calc new move
        eval_matrix = matrix - alpha * rand_vec 

        # calculate statistics
        wordsim = corr_CD_wordsim(eval_matrix, words, gold)
        bias = corr_CD_freq(eval_matrix, words, gold, freqs)
        log_bias = corr_CD_freq(eval_matrix, words, gold, logfreqs)
        isotropy, len_centroid = isotropyANDcentroid(eval_matrix)

        mean_sp = mean_scalar_proj(eval_matrix, centroid)
        cosine_sim_rand = np.dot(centroid, rand_vec) / (np.linalg.norm(centroid) * np.linalg.norm(rand_vec))

        # print statistics
        result_tuple = (rot, wordsim, bias, log_bias, isotropy, len_centroid, mean_sp, cosine_sim_rand)
        output = ''
        for entry in result_tuple:
            output += '{:.6f}\t'.format(entry)
        print(output.strip())

        # increment alpha, rounding cause python cant do maths...
        alpha = round(alpha + step, 2)

    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
