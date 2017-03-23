import os.path
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from scripts.utility import help_read_csv

def prep(data_directory):

	print 'Preparing IMDB data...\n'

	"""

	SETUP

	"""

	# Read in data and print shape
	data = help_read_csv(os.path.join(data_directory, 'movie_metadata.csv'))

	print '- Read in .csv file with {} rows and {} columns'.format(*data.shape)

	# Drop unneeded variables
	data.drop(['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'plot_keywords', 'aspect_ratio'], axis=1, inplace=True)

	print '- Dropped unneeded variables'

	# Extract IMDB movie id from link as `imdb_id`
	data['movie_imdb_link'] = data['movie_imdb_link'].astype('str')
	data['imdb_id'] = data.movie_imdb_link.str.extract('(tt[0-9]+)', expand = False)
	data['imdb_id'] = data['imdb_id'].str.extract('([0-9]+)', expand = False)
	data.drop(['movie_imdb_link'], axis=1, inplace=True)

	print '- Extracted IMDB movie ids as `imdb_id` from url in `movie_imdb_link`'


	# Drop duplicated movies
	data.drop_duplicates('imdb_id', inplace = True)
	data.reset_index(drop = True, inplace=True)

	print '- Duplicate movies removed'

	# 'movie_title', 'imdb_id'
	# should also be dropped from analyses, but are needed to merge with other data sets

	"""

	CATEGORICAL VARIABLES

	"""

	# Dummy code `language` to `english` = 1 (yes) or 0 (no)
	# Because english language dominates large proportion
	data['english'] = data['language'] == 'English'
	data.drop(['language'], axis=1, inplace=True)

	print '- Dummy coded `language` to boolean, `english`'

	# Dummy code `country` to two bool variables: `usa`, `uk`
	# US (particularly) and UK dominate. Other can be own category.
	data['usa'] = data['country'] == 'USA'
	data['uk']  = data['country'] == 'UK'
	data.drop(['country'], axis=1, inplace=True)

	print '- Dummy coded `country` to two boolean vars, `usa`, `uk`'

	# Dummy code color to 1 (color) or 0 (Black and white)
	data['color'] = data['color'] == 'Color'

	print '- Dummy coded `color` to boolean.'

	# Dummy code content ratings
	dummy_ratings = pd.get_dummies(data.content_rating.str.lower()).astype(int)
	data = pd.concat([data, dummy_ratings], axis=1, join='inner')
	data.drop(['content_rating'], axis=1, inplace=True)

	print '- Dummy coded `content_ratings` to multiple columns'

	# Dummy code genres
	dummy_genres = data.genres.str.lower().str.get_dummies(sep='|')
	data = pd.concat([data, dummy_genres], axis=1, join='inner')
	data.drop(['genres'], axis=1, inplace=True)

	print '- Dummy coded `genres` to multiple columns'

	"""

	MISSING VALUES

	"""


	float_vars = list(data.select_dtypes(include=['float64']))

	# For gross, missing values are probably 0s, so handle this first
	data.gross.fillna(0, inplace=True)

	# Handling missing values...
	data.fillna(data.mean()[float_vars], inplace=True)  # For float columns
	data.fillna(data.drop(float_vars, axis=1).mode(), inplace=True)  # For all others

	print '- Missing values imputed as mean/mode'

	"""

	CONTINUOUS VARIABLES

	"""

	# Distribution Transformations
	data.num_critic_for_reviews = np.sqrt(data.num_critic_for_reviews)
	data.gross = np.sqrt(data.gross)

	data.director_facebook_likes = np.log(data.director_facebook_likes + 1)
	data.actor_1_facebook_likes = np.log(data.actor_1_facebook_likes + 1)
	data.actor_2_facebook_likes = np.log(data.actor_2_facebook_likes + 1)
	data.actor_3_facebook_likes = np.log(data.actor_3_facebook_likes + 1)
	data.budget = np.log(data.budget)
	data.duration = np.log(data.duration)
	data.num_user_for_reviews = np.log(data.num_user_for_reviews)

	# An extreme option. See http://stackoverflow.com/questions/38151261/find-a-python-transformation-function-or-numpy-matrix-to-transform-skewed-normal
	#data.num_critic_for_reviews = norm.ppf((data.num_critic_for_reviews.rank() - .5) / len(data))


	print '- Distribution adjustments applied to selected float variables'


	# Normalize (0-1) all float columns
	data[float_vars] = data[float_vars].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

	print '- All float variables normalized to range between 0 to 1'


	"""

	END

	"""


	print '\nPrep of IMDB data completed.\n'
	return data


"""
Conduct PCA on float variables and replace with dimensions
"""


def pca_of_float_vars(data, n_components = None):

	float_vars = list(data.select_dtypes(include=['float64']))
	float_data = data[float_vars]

	if n_components is None:
		n_components = len(float_vars)

	pca = PCA(n_components=n_components)
	pca.fit(float_data)

	reduced_data = pca.transform(float_data)
	reduced_data = pd.DataFrame(reduced_data, columns=list('dimension_' + str(i) for i in xrange(n_components)))

	# Replace float variables with dimension results
	data = pd.concat([data.drop(float_vars, axis=1), reduced_data], axis=1, join='inner')

	return data, pca
