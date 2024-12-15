from naiveBayes import NaiveBayes

if __name__ == "__main__":
    print("Running Naive Bayes training")
    nb = NaiveBayes()
    nb.set_data_source('mushrooms.csv')
    nb.fit()