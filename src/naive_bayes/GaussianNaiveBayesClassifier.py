import pandas as pd
# import numpy as np
from typing import Self, List
from os import path
from sys import stderr
from sklearn.model_selection import train_test_split
from math import log, pi, sqrt
from numpy import exp

# TODO - Make descriptions 

class NormalDistribution:
    def __init__(self: Self, mean: float = None, variance: float = None):
        self.mean = mean
        self.variance = variance
    
    def get_value(self: Self, x: float):
        """
        Calculates the probability density function (PDF) value in point x.
        If mean or variance are None, it raises an error
        """
        if self.mean is None or self.variance is None:
            raise ValueError("Values not set for Gaussian distribution") 
        if self.variance == 0:
            return 0
        try:
            return exp(-(x-self.mean)**2/(2*self.variance))*(1/sqrt(2*pi*self.variance))
        except Exception as e:
            print(e)
            return 0
    def fit_values(self: Self, values: List[float]):
        """
        Fits the mean and variance based on the provided list of data points.
        """
        self.mean = sum(values)/len(values)
        squaredMean = sum([ element ** 2 for element in values ])/len(values)
        self.variance = squaredMean - self.mean ** 2

    def __str__(self: Self):
        return f"Mean of Gaussian: {self.mean}\nVariance of Gaussian: {self.variance}\n"

    def __repr__(self: Self):
        return str(self)

class GaussianNaiveBayesClassifier:
    def __init__(self: Self) -> None:
        """
        Constructor of the Naive Bayes class
        """
        # Stores the test dataset used for evaluating the model.
        self.testSet = None  
        # General-purpose dictionary, possibly used for internal mappings or configurations.
        self.dict = {}  
        # Stores the likelihood of each class for classification.
        self.classesLikelyhood = {}  
        # Keeps track of the count of each class in the training dataset.
        self.classesCount = {}  
        # Stores the main dataset as a DataFrame.
        self.df = None  
        # Fraction of the data to be used for training; the rest is for testing.
        self.testTrainSplit = 0.7  
        # Stores the size of the dataset (number of rows or total entries).
        self.dataSize = None  
        # A set of all unique class labels in the dataset.
        self.classes = set()  
        # Stores traits grouped by class, with trait values organized per class.
        self.conditionalTraitsList = {}  
        # Maps traits to their Gaussian distribution function for each class.
        self.conditionalProbabilities = {}  # 'trait': function
        # A list of names of all traits (features/columns) in the dataset.
        self.traitNames = []  
        # Smoothing parameter for probability calculations (e.g., Laplace smoothing).
        self.alpha = 1  
        # Stores the current directory of the script file.
        self.current_dir = path.dirname(__file__)  
        # Path to the dataset; defaults to the current directory.
        self.dataPath = self.current_dir  
        # Tracks anomalies or unusual cases encountered during processing.
        self.oddities = 0  

    def set_split_ratio(self: Self, ratio: float):
        self.testTrainSplit = ratio
        

    def set_data_source(self: Self,filename: str,separator: str = ',') -> None:
        """
        Sets the data source for the model to train on
        Arguments:
        filename: Name of the file with the extension
        separator: Defines a data separator in the file 
        Returns:
        Nothing
        """
        # Relative path to the data file, should be changed later so that it works in general
        self.relativePath = path.join('../../', 'data',filename)
        # Absolute (!) path to the data file
        self.dataPath = path.abspath(self.relativePath)
        self.separator = separator

    def read_data(self: Self) -> None:
        """
        Reads a csv file using pandas from the given filepath and saves it into self.df 
        """
        if not self.dataPath:
            print("First set the data source!")
            return
        try:
            dataframe = pd.read_csv(self.dataPath)
        except Exception as error:
            print("An error has occured while reading the file (check if the file exists)", file=stderr)
            raise error
        self.df = dataframe

    def dump_data(self: Self) -> None:
        """
        Prints the whole dataframe
        """
        print(self.df)

    def add_traits(self: Self):
        for traitName in self.trainSet.columns:
            self.traitNames.append(traitName)

    def add_class(self: Self, value: str):
        if value not in self.classes:
            self.classes.add(value)

    def add_class_value(self: Self, value: str) -> None:
        """
        Adds a new class to classesCount or incements the amount of the existing class by 1
        """
        self.add_class(value)
        if value not in self.classesCount:
            self.classesCount[value] = 1
        else:
            self.classesCount[value] += 1

    def calculate_classes_likelyhoods(self: Self) -> None:
        """
        Calculates likelyhoods of each class of all classes
        """
        for key,value in self.classesCount.items():
            self.classesLikelyhood[key] = value/self.dataSize

    def create_sets(self: Self):        
        for index, row in self.trainSet.iterrows():
            className = row[self.trainSet.columns[len(self.trainSet.columns)-1]]
            self.add_class_value(className)
            if className not in self.conditionalTraitsList:
                self.conditionalTraitsList[className] = {}
            for column in self.trainSet.columns[:len(self.trainSet.columns)-1]:
                if column not in self.conditionalTraitsList[className]:
                    self.conditionalTraitsList[className][column] = []
                self.conditionalTraitsList[className][column].append(row[column])
    
    def calculate_gaussians(self: Self):
        for className, values in self.conditionalTraitsList.items():
            if className not in self.conditionalProbabilities:
                self.conditionalProbabilities[className] = {}
            for trait,array in values.items():
                gaussian = NormalDistribution()
                gaussian.fit_values(array)
                self.conditionalProbabilities[className][trait] = gaussian
    
    def test_accuracy(self: Self, testSet: pd.DataFrame):
        self.testSet = testSet
        return self.test_local()
    
    def test_local(self: Self) -> float:
        """
        Returns accuracy of the test
        """
        correctGuesses = 0
        adjustmentGuesses = 0
        for rowIndex, test in self.testSet.iterrows():
            className, likelyhoodValue = self.predict(test[:len(test)-1])
            if className == test[len(test)-1]:
                correctGuesses += 1
                adjustmentGuesses += 1 # Didn't need adjustment
            else:
                predictions = self.predict_proba(test[:len(test)-1])
        accuracy = correctGuesses/len(self.testSet)
        adjustmentAccuracy = adjustmentGuesses/len(self.testSet)
        return accuracy
    
    def fit_test(self: Self):
        self.read_data()
        self.trainSet, self.testSet

    def fit(self: Self,trainSet):
        self.trainSet = trainSet
        self.fit_local()

    def set_train_set(self: Self):
        self.read_data()
        self.trainSet, self.testSet = train_test_split(self.df, test_size=self.testTrainSplit)

    def fit_local(self: Self):
        """
        Trains the model based on the given data
        Arguments:
        Nothing
        Returns:
        True - if the data has been analyzed and trained successfully
        False - if the training set is not configured properly
        Raises Error - if an error occured while reading the data
        """
        self.dataSize = self.trainSet.size//len(self.trainSet.columns)

        self.add_traits()
        self.create_sets()
        self.calculate_gaussians()
        self.calculate_classes_likelyhoods()

    def select_most_likely(self: Self, predictedProbabilities: dict[str,float]) -> tuple[str,float]:
        maxScore = -float('inf')
        maxClass = None
        for key, value in predictedProbabilities.items():
            if value > maxScore:
                maxScore = value
                maxClass = key
        return maxClass, maxScore

    def predict(self: Self, dataRow: tuple[str]) -> tuple[str,float]:
        """
        Predicts which value is most likely to appear
        Arguments:
        Nothing
        Returns:
         - A tuple with:
           > A string with class that has been chosen as the most likely
           > Probability of that class in the inverval [0,1]
        """
        predictedProbabilities = self.predict_proba(dataRow)
        return self.select_most_likely(predictedProbabilities)
  
    def predict_proba(self: Self, dataRow: tuple[float]) -> dict[str,float]:
        """
        Predicts which value is most likely to appear
        Arguments:
        Nothing
        Returns:
         - A list of probabilities of classification
        """
        classProbabilities = {}
        classScores = {}
        classPartialProbabilities = { element:[] for element in self.classes }
        for className in self.classes:
            resultProbability = 0
            accumulatedTraits = 1
            summedClassScore = 0
            for i,traitValue in enumerate(dataRow[:len(dataRow)-1]):
                conditionalLikelyhood = self.conditionalProbabilities[className][self.traitNames[i]].get_value(traitValue) 
                accumulatedTraits *= conditionalLikelyhood
                if not conditionalLikelyhood <= 0:
                    logarithmicConditionalScore = log(conditionalLikelyhood)
                else:
                    logarithmicConditionalScore = 0
                summedClassScore += logarithmicConditionalScore
            classLikelyhood = self.classesLikelyhood[className]
            # Should not be the case, but it could happen that the new class appears
            if not classLikelyhood <= 0:
                logClassScore = log(classLikelyhood)
            else:
                logClassScore = -float('inf')
            resultProbability = accumulatedTraits*classLikelyhood
            summedClassScore += logClassScore
            classProbabilities[className] = resultProbability
            classScores[className] = summedClassScore # Will be negative
        
        values = [ classProbabilities[key] for key in classProbabilities.keys() ]
        sumOfValues = sum(values)
        # Check for any anomalies where probabilities exceed 1
        for key in classProbabilities.keys():
            classProbabilities[key] = classProbabilities[key] / sumOfValues 
        
        values = [ exp(classScores[key]) for key in classScores.keys() ]
        sumOfValues = sum(values)        
        for key in classScores.keys():
            classScores[key] = exp(classScores[key]) / sumOfValues

        return classProbabilities
