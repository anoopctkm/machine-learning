import os.path
import pandas as pd
import numpy as np

from scripts.utility import help_read_csv


def prep(data_directory):

    print 'Preparing MovieLens data...\n'

    """

    SETUP

    """

    # Read in rating data
    data = help_read_csv(os.path.join(data_directory, 'rating.csv'))
    print '- Read in .csv file with {} rows and {} columns'.format(*data.shape)


    """

    END

    """

    print '\nPrep of MovieLens data completed.\n'
    return data