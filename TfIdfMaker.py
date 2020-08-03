import pandas as pd
import numpy as np
import pickle

# class to create the tf-idf matrix with the values for each document
class TfIdfMaker:
    def __init__(self, file_name):
        self.file_name = file_name
        self.vocabulary_set = set()
        self.terms_and_movies_frequencies = None
        self.c = 0


    # get the vocabulary in the summaries
    def get_vocabulary(self, summary):
        self.vocabulary_set = self.vocabulary_set.union(set(summary))
        return summary
    

    # fill out the frequencies
    def get_frequencies(self, row):
        # get necessary information to fill in frequences
        summary = row["Summary"]
        name = row["Name"]

        # fill into the other table
        for word in summary:
            self.terms_and_movies_frequencies.loc[[word], [name]] += 1
        
        self.c += 1
        if self.c % 50 == 0:
            print(self.c)

        return row

    
    # creates the frequencies for the tfidf matrix for the data provided 
    def create_frequencies(self):
        # read in the data
        self.id_name_summary = pd.read_pickle(self.file_name)

        # obtain the vocabulary
        self.id_name_summary["Summary"] = self.id_name_summary["Summary"].apply(self.get_vocabulary)
        self.vocabulary_list = list(self.vocabulary_set)

        # create dataframe with rows as terms and columns as movie names
        columns = self.id_name_summary["Name"]
        rows = pd.Index(self.vocabulary_list)
        self.terms_and_movies_frequencies = pd.DataFrame(0, index=rows, columns = columns)
        self.terms_and_movies_frequencies = self.terms_and_movies_frequencies.loc[:,~self.terms_and_movies_frequencies.columns.duplicated()]

        print(self.terms_and_movies_frequencies)

        # get the frequencies
        self.id_name_summary.apply(self.get_frequencies, axis = 1)

        print(self.terms_and_movies_frequencies)

        # save as a pickle
        self.terms_and_movies_frequencies.to_pickle("terms_and_movies.pkl")
    

    # calculate the tf_idf values
    def create_tf_idf(self):
        # read in the data
        self.terms_and_movies_frequencies = pd.read_pickle("terms_and_movies.pkl")

        # get the tf matrix
        self.terms = self.terms_and_movies_frequencies.copy()
        self.sums_rows = self.terms.sum(axis = 0, skipna = True)
        self.tf = self.terms/self.sums_rows

        # get the idf matrix
        self.binary_terms = self.terms.copy()
        self.binary_terms[self.binary_terms > 0] = 1
        self.sum_cols = self.binary_terms.sum(axis = 1, skipna = True)
        self.idf = 219/self.sum_cols
        self.idf = np.log(self.idf)
        # print(self.idf)
        pd.set_option("max_rows", None)
        
        
        print(self.tf.shape)
        print(self.idf.shape)
        # print(self.tf)
        # print(self.idf)
        self.tf_idf = self.tf.divide(self.idf, axis=0)
        # print(self.tf_idf)


        # save as pickle
        self.tf_idf.to_pickle("tf_idf.pkl")

        