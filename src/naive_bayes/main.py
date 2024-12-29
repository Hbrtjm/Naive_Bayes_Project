from MultinomialNaiveBayesClassifier import MultinomialNaiveBayesClassifier
from GaussianNaiveBayesClassifier import GaussianNaiveBayesClassifier
if __name__ == "__main__":
    print("Running Naive Bayes training")
    nbg = GaussianNaiveBayesClassifier()
    nbg.set_data_source("iris.csv")
    nbg.fit()
    
    # nbc = MultinomialNaiveBayesClassifier()
    # nbc.set_data_source('mushrooms.csv')
    # nbc.fit()
    # data = ('b','y','n','f','p','f','w','b','n','t','c','s','s','w','w','p','w','o','e','k','s','m')
    # className, probability = nbc.predict(data)
    # print(f"Object classified as \"{className}\" with score of {probability}")
    # print(f"Oddities {nbc.oddities}")
        