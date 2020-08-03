import pandas as pd
import pickle
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

# class that will extract and clean the data needed from the file provided
class DataCleaner:
    def __init__(self, file_name):
        self.file_name = file_name
        self.c = 0
        self.summary_lengths = []
    

    # clean up the summary
    def clean_summary(self, summary):
        # lowercase
        lower_summary = summary.lower()

        # words in sentence
        words = word_tokenize(lower_summary)

        # remove stop words and werid punctions and do stemming
        ps = PorterStemmer()
        filtered_summary = [ps.stem(w) for w in words if w not in stopwords.words("english") and (w.isalnum())]

        if self.c % 50 == 0:
            print(self.c)

        self.c += 1
        self.summary_lengths.append(len(filtered_summary))

        return filtered_summary


    # overall function to read and clean the data
    def clean_data(self):
        # read all the summaries in
        id_to_summaries = pd.read_csv("plot_summaries.txt", sep="\t", header=None, names=["Id", "Summary"])
        
        # read in all the metadata
        all_data = pd.read_csv("movie.metadata.tsv", sep="\t", header=None, names=["Wiki_Id", "Free_Id", "Name", "Release", "Revenue", "Runtime", "Languages", "Countries", "Genres"])

        # get only the id to names
        id_to_names = all_data[["Wiki_Id", "Name"]]

        # do an inner join to get the ids corresponding to names and summaries
        id_name_summary = pd.merge(id_to_summaries, id_to_names, left_on='Id', right_on='Wiki_Id', how="inner").drop("Wiki_Id", axis=1).tail(2000)

        # remove all stop words in all the plots
        id_name_summary["Summary"] = id_name_summary["Summary"].apply(self.clean_summary)

        id_name_summary = id_name_summary.loc[:,~id_name_summary.columns.duplicated()]
        
        # get approximation for number in summary
        length_counts = Counter(self.summary_lengths)
        print(length_counts.most_common(30))
        
        # create a limit based on number of words
        limited_id_name_summary = id_name_summary[(id_name_summary.Summary.apply(len) >= 200) & (id_name_summary.Summary.apply(len) <= 300)]

        print(limited_id_name_summary)

        # convert them to pickles
        id_name_summary.to_pickle("id_name_summary.pkl")
        limited_id_name_summary.to_pickle("limited_id_name_summary.pkl")



