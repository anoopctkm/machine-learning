import os.path
import zipfile

def extract_all(data_directory):
    folders_to_unzip = ["imdb-5000-movie-dataset.zip", "movielens-20m-dataset.zip"]
    for zip_file in folders_to_unzip:
        print "Extracting all from " + zip_file + "...\t",
        path_to_zip_file = os.path.join(data_directory, zip_file)
        zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
        zip_ref.extractall(data_directory)
        zip_ref.close()
        print "DONE"
