from naiveBayes import NaiveBayes

if __name__ == "__main__":
    print("Running Naive Bayes training")
    nb = NaiveBayes()
    nb.set_data_source('mushrooms.csv')
    nb.fit()
    data = ('b','y','n','f','p','f','w','b','n','t','c','s','s','w','w','p','w','o','e','k','s','m')
    className, probability = nb.predict(data)
    print(f"Object classified as \"{className}\" with probability {probability}")