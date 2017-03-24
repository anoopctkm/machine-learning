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

    # Bind IMDB ID using link data
    link = help_read_csv(os.path.join(data_directory, 'link.csv'))
    link = link.drop('tmdbId', axis = 1)
    data = data.join(link.set_index('movieId'), on = 'movieId', how = 'left')
    data = data.rename(columns={'imdbId': 'imdb_id'}).drop('movieId', axis=1)
    del link

    print '- Bind IMDB ID as `imdb_id` using link.csv'

    """

    END

    """

    print '\nPrep of MovieLens data completed.\n'
    return data