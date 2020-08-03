import pandas as pd
import numpy as np
import pickle
import sklearn.cluster

# class to do the clustering
class KMeansClusterer:
    def __init__(self, file_name):
        self.file_name = file_name
    
    # find the clusters and do the labels
    def find_clusters(self):
        # read the tf idf values
        self.tf_idf = pd.read_pickle(self.file_name)
        self.tf_idf_values = self.tf_idf.values

        # do k means clustering
        km = sklearn.cluster.KMeans(n_clusters=3)
        km.fit(self.tf_idf_values)

        # get te different labels
        labels = km.labels_

        # put into df again
        results = pd.DataFrame([self.tf_idf.index, labels]).T

        pd.set_option("max_rows", None)
        print(results)

