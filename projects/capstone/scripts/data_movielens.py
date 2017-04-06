import os.path
import pandas as pd
import numpy as np

from scripts.utility import help_read_csv


def prep(data_directory, imdb_ids):

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


    # Retain only movies appearing in the IMDB data set
    data = data[data['imdb_id'].isin(imdb_ids)]

    print '- Movies not appearing in IMDB dataset dropped'


    # Retain users who have made k or more ratings
    # - helps to have only users with enough info as well as reducing the data set size (which is massive)
    k = 1000
    data = data.groupby('userId').filter(lambda x: len(x) >= k)

    print '- Drop users who made fewer than ' + str(k) + ' ratings'

    """

    END

    """

    data.reset_index(drop=True, inplace=True)
    print '\nPrep of MovieLens data completed.\n'
    return data