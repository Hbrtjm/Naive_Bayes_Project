import pandas as pd
from typing import Self, List
from os import path
from sys import stderr
from sklearn.model_selection import train_test_split
from math import log
from numpy import exp

class MultinomialNaiveBayesClassifier:

    def __init__(self: Self) -> None:
        """
        Constructor of the Naive Bayes class
        """
        # Train-test split ratio, in case the data is read from the model
        self.testTrainSplit = 0.7
        # Total number of data points
        self.dataSize = None
        # DataFrame containing the dataset
        self.df = None
        # Unique classes in the dataset
        self.classes = set()
        # Traits to exclude from analysis, should be by the index
        self.excludedTraits = set()
        # Dictionary of traits and their possible values
        self.traits = {}
        # Names of all traits
        self.traitNames = []
        # Conditional probabilities of traits given classes
        self.traitsConditionalLikelyhood = {}
        # Count of each trait's occurrences
        self.traitsCount = {}
        # Likelihood of each trait value
        self.traitsLikelyhood = {}
        # Conditional count of traits per class
        self.traitsConditionalCount = {}
        # Count of each class
        self.classesCount = {}
        # Likelihood of each class
        self.classesLikelyhood = {}
        # Smoothing factor for Naive Bayes
        self.alpha = 1
        # Current directory path
        self.current_dir = path.dirname(__file__)
        # Path to the dataset
        self.dataPath = self.current_dir
        # Counter for unusual probabilities
        self.oddities = 0

    def set_split_ratio(self: Self, ratio: float):          
        """
        Sets the ratio for splitting the dataset into training and testing sets.
        """
        self.testTrainSplit = ratio
        
    def set_data_source(self: Self,filename: str,separator: str = ',') -> None:

        """
        Sets the data source for the model to train on.
        
        Arguments:
        - filename: Name of the file with the extension
        - separator: Defines the delimiter used in the file
        """
        # Relative path to the data file, should be changed later so that it works in general
        self.relativePath = path.join('../../', 'data',filename)
        # Absolute (!) path to the data file
        self.dataPath = path.abspath(self.relativePath)
        self.separator = separator
    
    def add_trait_names(self: Self):
        """
        Initializes the traits dictionary with column names (except the class column).
        """
        for column in self.trainSet[1:]:
            self.traits[column] = set()

    def add_traits(self: Self):
        """
        Updates the traits dictionary with all of the available traits, especially for the missing traits
        """
        for traitName in self.trainSet.columns[1:]:
            self.traitNames.append(traitName)
        for index, row in self.trainSet.iterrows():
            for traitName in self.trainSet.columns[1:]:
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

    def add_trait_values(self: Self, row) -> None:
        """
        For each trait function counts the occurence of its values in the dataset
        """
        # Skip the class column
        for column in self.trainSet.columns[1:]:
            if column not in self.traitsCount:
                self.traitsCount[column] = { row[column]:1 }
                continue
            if row[column] not in self.traitsCount[column]:
                self.traitsCount[column][row[column]] = 1
                continue
            self.traitsCount[column][row[column]] += 1

    def add_conditional_likelyhood(self: Self, row) -> None:
        """
        Counts the conditional occurrences of trait values for each class.
        """
        objectClass = row[self.trainSet.columns[0]] 
        if objectClass not in self.traitsConditionalCount:
            self.traitsConditionalCount[objectClass] = {}
        for column in self.trainSet.columns[1:]:
            if column not in self.traitsConditionalCount[objectClass]:
                self.traitsConditionalCount[objectClass][column] = { row[column]: 1 }
                continue
            if row[column] not in self.traitsConditionalCount[objectClass][column]:
                self.traitsConditionalCount[objectClass][column][row[column]] = 1
                continue
            self.traitsConditionalCount[objectClass][column][row[column]] += 1
    
    def add_alpha_to_traits(self: Self) -> None:
        """
        Applies Laplace smoothing to trait counts to avoid zero probabilities.
        """
        for traitName in self.traitNames: # Can be done using keys, but it's for clarity
            for trait in self.traits[traitName]:
                if trait not in self.traitsCount[traitName]:
                    self.traitsCount[traitName][trait] = self.alpha
                    continue
                self.traitsCount[traitName][trait] += self.alpha

    def add_alpha_to_conditional_traits(self: Self) -> None:
        """
        Applies Laplace smoothing to conditional trait counts.
        """
        for className in self.classes:
            for traitName in self.traitNames: # Can be done using keys, but it's for clarity
                for trait in self.traits[traitName]:
                    if trait not in self.traitsConditionalCount[className][traitName]:
                        self.traitsConditionalCount[className][traitName][trait] = self.alpha
                        continue
                    self.traitsConditionalCount[className][traitName][trait] += self.alpha
    
    def calculate_trait_likelyhoods(self: Self) -> None:
        """
        Calculates likelyhoods of all values of all traits
        """
        for column in self.trainSet.columns[1:]:
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
            
    def calculate_conditional_likelyhoods(self: Self) -> None:
        """
        Calculates conditional likelyhood for each class given the trait
        """
        for className in self.classes:
            if className not in self.traitsConditionalLikelyhood:
                self.traitsConditionalLikelyhood[className] = {}
            for column in self.trainSet.columns[1:]:
                # Each column has to have a value in this case, it will throw an error
                for key,value in self.traitsConditionalCount[className][column].items():
                    if column not in self.traitsConditionalLikelyhood[className]:
                        self.traitsConditionalLikelyhood[className][column] = {}
                    self.traitsConditionalLikelyhood[className][column][key] = value / (self.dataSize+self.alpha*len(self.traitsConditionalLikelyhood[className][column]))
    
    def read_data(self: Self) -> None:
        """
        Reads data from the self.dataPath
        """
        if not self.dataPath:
            print("First set the data source!")
            return False
        try:
            dataframe = pd.read_csv(self.dataPath)
        except Exception as error:
            print("An error has occured while reading the file (check if the file exists)", file=stderr)
            raise error
        self.df = dataframe

    def test_accuracy(self: Self, testSet):
        self.testSet = testSet
        return self.test_local()
    
    def test_local(self: Self):
        """
        Tests the model using the pd.DataFrame specified in self.testSet
        """
        correctGuesses = 0
        adjustmentGuesses = 0
        for rowIndex, test in self.testSet.iterrows():
            className, likelyhoodValue = self.predict(test[1:])
            if className == test[0]:
                correctGuesses += 1
                adjustmentGuesses += 1
            else:
                predictions = self.predict_proba(test[1:])
                newPredictions = self.make_mushroom_safe(predictions)
                newClassName, _ = self.select_most_likely(newPredictions)
                if newClassName == test[0]:
                    adjustmentGuesses += 1
        accuracy = correctGuesses / len(self.testSet)
        adjustmentAccuracy = adjustmentGuesses / len(self.testSet)
        return accuracy
    
    
    def fit(self: Self, trainSet: pd.DataFrame):
        self.trainSet = trainSet
        self.fit_local()

    def set_train_set(self: Self):
        self.read_data()
        self.trainSet, self.testSet = train_test_split(self.df, test_size=self.testTrainSplit)


    def fit_local(self: Self) -> None:
        """
        Internal method for training the model. Computes counts and likelihoods.
        """
        
        # =============== Read data, set train and test sets ===============
        
        self.add_trait_names()
        self.add_traits()
        self.dataSize = self.trainSet.size//len(self.trainSet.columns)
        # Set the class column name (usually 'class')
        classColumnName = self.trainSet.columns[0]
        # Count classes and traits
        for index, row in self.trainSet.iterrows():
            self.add_class_value(row[classColumnName])
            self.add_trait_values(row)
            self.add_conditional_likelyhood(row)
        
        self.add_alpha_to_traits()
        self.add_alpha_to_conditional_traits()

        # =============== Calculate likelyhoods ===============
        
        self.calculate_trait_likelyhoods()
        self.calculate_classes_likelyhoods()
        self.calculate_conditional_likelyhoods()
    
    def make_mushroom_safe(self: Self, predictions: dict[str,float]):
        # ONLY FOR MUSHROOM PREDICTOR
        # if 'e' in predictions:
        #     if predictions['e'] < 0.2: # Should not be considered safe to eat
        #         predictions['e'] = 0
        return predictions

    def select_most_likely(self: Self, predictedProbabilities: dict[str,float]) -> tuple[str,float]:
        """
        Selects the class with the highest score.
        """
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
  
    def predict_proba(self: Self, dataRow: tuple[str]) -> dict[str, float]:
        """
        Predicts the probabilities of each class for a given data row.
        Arguments:
        - dataRow: A tuple containing the values of the features for a single data instance.
        
        Returns:
        - A dictionary containing the log scores of each class as keys and their respective 
          log-probabilities as values.
        """
        classProbabilities = {}
        classScores = {}
        classPartialProbabilities = {element: [] for element in self.classes}

        # Iterate over each class to calculate probabilities and scores.
        for className in self.classes:
            resultProbability = 0
            accumulatedTraits = 1
            summedClassScore = 0

            # Iterate over each feature (trait) value in the data row.
            for i, traitValue in enumerate(dataRow):
                # Check if the trait value exists in the conditional likelihood dictionary for the current class.
                if traitValue not in self.traitsConditionalLikelyhood[className][self.traitNames[i]]:
                    # Assign a default value of 1 if the trait value is missing. That will result in log score being 0 - so no change for the score
                    conditionalLikelyhood = 1
                    traitsLikelyhood = 1
                else:
                    # Retrieve the conditional likelihood and the overall likelihood for the current trait value.
                    conditionalLikelyhood = self.traitsConditionalLikelyhood[className][self.traitNames[i]][traitValue]
                    traitsLikelyhood = self.traitsLikelyhood[self.traitNames[i]][traitValue]

                # Update the accumulated product of conditional likelihoods for this class.
                accumulatedTraits *= conditionalLikelyhood

                # Log the intermediate computation for debugging.
                classPartialProbabilities[className].append(f"{conditionalLikelyhood} / {conditionalLikelyhood}")

                # Compute the logarithmic contributions for better numerical stability.
                logarithmicConditionalScore = log(conditionalLikelyhood)
                logarithmicTraitsScore = log(traitsLikelyhood)
                summedClassScore += logarithmicConditionalScore - logarithmicTraitsScore

            # Retrieve the prior probability (likelihood) of the class.
            classLikelyhood = self.classesLikelyhood[className]
            logClassScore = log(classLikelyhood)

            # Combine the accumulated traits and class prior to compute the final probability for this class.
            resultProbability = accumulatedTraits * classLikelyhood

            # Update the summed class score by adding the log of the class likelihood.
            summedClassScore += logClassScore

            # Store the final probabilities and log scores for the class.
            classProbabilities[className] = resultProbability # In case it's needed
            classScores[className] = summedClassScore  # The log score will usually be negative.

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
