from DataCleaner import DataCleaner
from TfIdfMaker import TfIdfMaker
from KMeansClusterer import KMeansClusterer

if __name__ == "__main__":
    # create the instance that takes in the file name to read data and clean from
    data_cleaner = DataCleaner("plot_summaries.txt")
    data_cleaner.clean_data()

    tf_idf_maker = TfIdfMaker("limited_id_name_summary.pkl")
    tf_idf_maker.create_frequencies()
    tf_idf_maker.create_tf_idf()

    k_means_clusterer = KMeansClusterer("tf_idf.pkl")
    k_means_clusterer.find_clusters()