import os.path
import pandas as pd
from scripts.utility import help_read_csv

def prep(data_directory):

	# Read in data and print shape
	data = help_read_csv(os.path.join(data_directory, 'movie_metadata.csv'))

	# Drop unneeded variables
	data.drop(['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'plot_keywords'], axis=1, inplace=True)

	# 'movie_title', 'movie_imdb_link'
	# should also be dropped eventually, but are needed to merge with other data sets

	return data
