from DataCleaner import DataCleaner


if __name__ == "__main__":
    # create the instance that takes in the file name to read data and clean from
    data_cleaner = DataCleaner("plot_summaries.txt")
    data_cleaner.clean_data()



