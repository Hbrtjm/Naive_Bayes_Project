import pandas as pd
# import numpy as np
from typing import Self, List
from os import path
from sys import stderr
from sklearn.model_selection import train_test_split
from math import log

class MultinomialNaiveBayesClassifier:

    def __init__(self: Self) -> None:
        """
        Constructor of the Naive Bayes class
        """
        self.testTrainSplit = 0.2
        self.dataSize = None
        self.df = None
        self.classes = set()
        self.traits = {}
        self.traitNames = []
        self.traitsConditionalLikelyhood = {}
        self.traitsCount = {}
        self.traitsLikelyhood = {}
        self.traitsConditionalCount = {}
        self.classesCount = {}
        self.classesLikelyhood = {}
        self.alpha = 1
        self.current_dir = path.dirname(__file__)
        self.dataPath = self.current_dir
        self.oddities = 0
    
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
    
    def add_trait_names(self: Self, df: pd.DataFrame):
        for column in df.columns[1:]:
            self.traits[column] = set()

    def add_traits(self: Self, df: pd.DataFrame):
        """
        Updates the traits dictionary with all of the available traits, especially for the missing traits
        """
        for traitName in df.columns[1:]:
            self.traitNames.append(traitName)
        for index, row in df.iterrows():
            for traitName in df.columns[1:]:
                self.traits[traitName].add(row[traitName])

    def add_class_value(self: Self, value: str) -> None:
        """
        Adds a new class to classesCount or incements the amount of the existing class by 1
        """
        if value not in self.classes:
            self.classes.add(value)
        if value not in self.classesCount:
            self.classesCount[value] = 1
        else:
            self.classesCount[value] += 1

    def add_trait_values(self: Self, df: pd.DataFrame, row) -> None:
        """
        For each trait function counts the occurence of its values in the dataset
        """
        # Skip the class column
        for column in df.columns[1:]:
            if column not in self.traitsCount:
                self.traitsCount[column] = { row[column]:1 }
                continue
            if row[column] not in self.traitsCount[column]:
                self.traitsCount[column][row[column]] = 1
                continue
            self.traitsCount[column][row[column]] += 1

    def add_conditional_likelyhood(self: Self, df: pd.DataFrame, row) -> None:
        objectClass = row[df.columns[0]] 
        if objectClass not in self.traitsConditionalCount:
            self.traitsConditionalCount[objectClass] = {}
        for column in df.columns[1:]:
            if column not in self.traitsConditionalCount[objectClass]:
                self.traitsConditionalCount[objectClass][column] = { row[column]:1 }
                continue
            if row[column] not in self.traitsConditionalCount[objectClass][column]:
                self.traitsConditionalCount[objectClass][column][row[column]] = 1
                continue
            self.traitsConditionalCount[objectClass][column][row[column]] += 1
    
    def add_alpha_to_traits(self: Self) -> None:
        for traitName in self.traitNames: # Can be done using keys, but it's for clarity
            for trait in self.traits[traitName]:
                if trait not in self.traitsCount[traitName]:
                    self.traitsCount[traitName][trait] = self.alpha
                    continue
                self.traitsCount[traitName][trait] += self.alpha

    def add_alpha_to_conditional_traits(self: Self) -> None:
        for className in self.classes:
            for traitName in self.traitNames: # Can be done using keys, but it's for clarity
                for trait in self.traits[traitName]:
                    if trait not in self.traitsConditionalCount[className][traitName]:
                        self.traitsConditionalCount[className][traitName][trait] = self.alpha
                        continue
                    self.traitsConditionalCount[className][traitName][trait] += self.alpha
    
    def calculate_trait_likelyhoods(self: Self, df: pd.DataFrame) -> None:
        """
        Calculates likelyhoods of all values of all traits
        """
        for column in df.columns[1:]:
            # Each column has to have a value in this case, it will throw an error
            for key,value in self.traitsCount[column].items():
                if column not in self.traitsLikelyhood:
                    self.traitsLikelyhood[column] = {}
                self.traitsLikelyhood[column][key] = value/(self.dataSize+self.alpha*len(self.traitsLikelyhood[column])) 

    def calculate_classes_likelyhoods(self: Self) -> None:
        """
        Calculates likelyhoods of each class of all classes
        """
        for key,value in self.classesCount.items():
            self.classesLikelyhood[key] = value/self.dataSize
            print(f"{value} / {self.dataSize}")

    def calculate_conditional_likelyhoods(self: Self, df: pd.DataFrame) -> None:
        for className in self.classes:
            if className not in self.traitsConditionalLikelyhood:
                self.traitsConditionalLikelyhood[className] = {}
            for column in df.columns[1:]:
                # Each column has to have a value in this case, it will throw an error
                for key,value in self.traitsConditionalCount[className][column].items():
                    if column not in self.traitsConditionalLikelyhood[className]:
                        self.traitsConditionalLikelyhood[className][column] = {}
                    # self.traitsConditionalLikelyhood[className][column][key] = value/(self.classesCount[className]+self.alpha*len(self.traitsConditionalLikelyhood[className][column]))
                    self.traitsConditionalLikelyhood[className][column][key] = value/(self.dataSize+self.alpha*len(self.traitsConditionalLikelyhood[className][column]))

    def read_data(self: Self) -> None:
        if not self.dataPath:
            print("First set the data source!")
            return False
        try:
            dataframe = pd.read_csv(self.dataPath)
        except Exception as error:
            print("An error has occured while reading the file (check if the file exists)", file=stderr)
            raise error
        self.df = dataframe
    
    def fit(self: Self) -> None:
        """
        Trains the model based on the given data
        Arguments:
        Nothing
        Returns:
        True - if the data has been analyzed and trained successfully
        False - if the training set is not configured properly
        Raises Error - if an error occured while reading the data
        """
        
        # =============== Read data, set train and test sets ===============
        
        self.read_data()
        trainSet, testSet = train_test_split(self.df, test_size=self.testTrainSplit)
        self.add_trait_names(trainSet)
        self.add_traits(self.df)
        self.dataSize = trainSet.size//len(trainSet.columns)
        # Set the class column name (usually 'class')
        classColumnName = trainSet.columns[0]
        # Count classes and traits
        for index, row in trainSet.iterrows():
            self.add_class_value(row[classColumnName])
            self.add_trait_values(trainSet,row)
            self.add_conditional_likelyhood(trainSet,row)
        
        self.add_alpha_to_traits()
        self.add_alpha_to_conditional_traits()

        # =============== Calculate likelyhoods ===============
        
        self.calculate_trait_likelyhoods(trainSet)
        self.calculate_classes_likelyhoods()
        self.calculate_conditional_likelyhoods(trainSet)

        # =============== Test after training ===============
        
        print("Testing after training")
        correctGuesses = 0
        adjustmentGuesses = 0
        for rowIndex, test in testSet.iterrows():
            # print(test)
            className, likelyhoodValue = self.predict(test[1:])
            if className == test[0]:
                correctGuesses += 1
                adjustmentGuesses += 1 # Didn't need adjustment
            else:
                predictions = self.predict_proba(test[1:])
                # print(f"Wrong predictions {predictions} expected {test[0]} but gotten {className}")
                newPredictions = self.make_mushroom_safe(predictions)
                newClassName, _ = self.select_most_likely(newPredictions)
                if newClassName == test[0]:
                    adjustmentGuesses += 1
        accuracy = correctGuesses/len(testSet)
        adjustmentAccuracy = adjustmentGuesses/len(testSet)
        print(f"Accuracy of {accuracy} accuracy after safety adjsutment {adjustmentAccuracy}") 

        # The model has been fit successfully
        return True
    
    def make_mushroom_safe(self: Self, predictions: dict[str,float]):
        # ONLY FOR MUSHROOM PREDICTOR
        if 'e' in predictions:
            if predictions['e'] < 0.2: # Should not be considered safe to eat
                predictions['e'] = 0
        return predictions

    def select_most_likely(self: Self, predictedProbabilities: dict[str,float]) -> tuple[str,float]:
        maxProb = -float('inf')
        maxClass = None
        for key in predictedProbabilities.keys():
            value =  predictedProbabilities[key] 
            if value > maxProb:
                maxProb = value
                maxClass = key
        return maxClass, maxProb

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
  
    def predict_proba(self: Self,dataRow: tuple[str]) -> dict[str,float]:
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
            for i,trait in enumerate(dataRow):
                # One-liner to be split up
                
                conditionalLikelyhood = self.traitsConditionalLikelyhood[className][self.traitNames[i]][trait] 
                traitsLikelyhood = self.traitsLikelyhood[self.traitNames[i]][trait]
                accumulatedTraits *= conditionalLikelyhood
                classPartialProbabilities[className].append(f"{self.traitsConditionalLikelyhood[className][self.traitNames[i]][trait]} / {self.traitsLikelyhood[self.traitNames[i]][trait]}")
                
                logarithmicConditionalScore = log(conditionalLikelyhood)
                logarithmicTraitsScore = log(traitsLikelyhood)
                summedClassScore += logarithmicConditionalScore - logarithmicTraitsScore
            classLikelyhood = self.classesLikelyhood[className]
            logClassScore = log(classLikelyhood)
            resultProbability = accumulatedTraits*classLikelyhood
            summedClassScore += logClassScore
            classProbabilities[className] = resultProbability
            classScores[className] = summedClassScore # Will be negative
        printClass = False
        for key in classProbabilities.keys():
            if classProbabilities[key] > 1:
                self.oddities += 1
                # print(f"Weird probability... To be verified")
                # print(classProbabilities)
                # printClass = True
        if printClass:
            for key in classProbabilities.keys():
                print(f"{key}: {classPartialProbabilities[key]}")
        return classScores
