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
    with open(filePath, 'r', encoding='utf-8') as f_in:
        names = f_in.readline().strip().split('\t')
        datalines = [line.strip().split('\t') for line in f_in]
    # create data structure
    data = [[float(entry[i]) for entry in datalines] for i,name in enumerate(names)]

    fig, ax1 = plt.subplots(dpi=200)
    for i,name in enumerate(names):
        if i == 0:
            continue
        ax1.plot(data[0], data[i], label=name)
    ax1.set_title(filePath.split('/')[-1] + ', ' + title)
    plt.xlabel(name[0])
    plt.legend(loc='upper right')
    plt.savefig(outname)

    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
