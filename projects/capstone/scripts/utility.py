import pandas as pd


def help_read_csv(path_to_csv):
	data = None
	try:
		data = pd.read_csv(path_to_csv)
		print path_to_csv + " read with {} rows and {} columns.".format(*data.shape)
	except:
		print "Dataset could not be loaded. Is the dataset missing?"
		
	return data


def count_plot(series):
	series.value_counts().plot(kind = 'bar')


def float_scatter_matrix(df):
	pd.scatter_matrix(df.select_dtypes(include=['float64']), alpha=0.2, diagonal='kde')

