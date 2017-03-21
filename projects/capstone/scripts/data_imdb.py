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

	# Dummy code `language` to `english` = 1 (yes) or 0 (no)
	# Because english language dominates large proportion
	data['english'] = data['language'] == 'English'
	data.drop(['language'], axis=1, inplace=True)

	# Dummy code `country` to two bool variables: `usa`, `uk`
	# US (particularly) and UK dominate. Other can be own category.
	data['usa'] = data['country'] == 'USA'
	data['uk']  = data['country'] == 'UK'
	data.drop(['country'], axis=1, inplace=True)

	# Dummy code color to 1 (color) or 0 (Black and white)
	data['color'] = data['color'] == 'Color'

	# Dummy code content ratings
	dummy_ratings = pd.get_dummies(data.content_rating.str.lower())
	data = pd.concat([data, dummy_ratings], axis=1, join='inner')
	data.drop(['content_rating'], axis=1, inplace=True)

	# Dummy code genres
	dummy_genres = data.genres.str.lower().str.get_dummies(sep='|')
	data = pd.concat([data, dummy_genres], axis=1, join='inner')
	data.drop(['genres'], axis=1, inplace=True)


	return data
