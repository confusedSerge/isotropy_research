import time
import logging
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt


def main():
    # Get the arguments
    args = docopt("""

    Usage:
        fast_plot.py <filePath> <outname> <title>

    Arguments:
        <filePath> = path to the scores
        <outname> = name for plots
        <title> = title
    """)

    filePath = args['<filePath>']
    outname = args['<outname>']
    title = args['<title>']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # read file
    x = []
    y = []
    with open(filePath, 'r', encoding='utf-8') as f_in:
        names = f_in.readline().strip().split('\t')
        for line in f_in.readlines():
            i = line.split('\t')
            x.append(float(i[0]))
            y.append(float(i[1]))
    
    fig, ax1 = plt.subplots(dpi=200)

    ax1.scatter(x, y)

    ax1.set_title(filePath.split('/')[-1] + ', ' + title)
    ax1.set_xlim([0,2])
    ax1.set_ylim([-1,1])
    plt.xlabel('delta(logfreq)')
    plt.ylabel('delta(scalarprojection)')
    plt.savefig(outname)

    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
