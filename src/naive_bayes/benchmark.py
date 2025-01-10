from MultinomialNaiveBayesClassifier import MultinomialNaiveBayesClassifier
from GaussianNaiveBayesClassifier import GaussianNaiveBayesClassifier
from statistics import mean

def run_tests(GivenClass,testSetName,modelName):
    # Clear the benchmark file
    with open('benchmark.txt','w') as dataFile:
        dataFile.write("")
    step = 0.1
    steps = int(1/step)
    testCases = [ i*step for i in range(1,steps) ]
    trials = 6
    print("Testing Gaussian Naive Bayes Classifier")
    for ratio in testCases:
        accuracies = []
        for _ in range(trials):
            nb = GivenClass()
            nb.set_data_source(testSetName)
            nb.set_split_ratio(ratio)
            nb.fit()
            accuracies.append(nb.test())
        print(f"For test/train split ratio {ratio}:\n\tAverage accuracy:\t{mean(accuracies)}\n\tMax accuracy:\t\t{max(accuracies)}\n\tMin accuracy: \t\t{min(accuracies)}\n")
        with open('benchmark.txt','+a') as dataFile:
            dataFile.write(f"For test/train split ratio {ratio}:\n\tAverage accuracy:\t{mean(accuracies)}\n\tMax accuracy:\t\t{max(accuracies)}\n\tMin accuracy: \t\t{min(accuracies)}\n")
    print("Testing Multinomial Naive Bayes Classifier")

def main():
    testClasses = [(MultinomialNaiveBayesClassifier,"mushrooms.csv","MNBC"),(GaussianNaiveBayesClassifier,"iris.csv","GNBC")]

    for testClass, setName, model in testClasses:
        run_tests(testClass,setName,model) 

if __name__ == "__main__":
    main()