import pandas as pd
import numpy as np
from typing import Self, List
from os import path
from sys import stderr
class NaiveBayes:
    def __init__(self: Self) -> None:
        """
        Constructor of the Naive Bayes class
        """
        self.dataSize = None
        self.traitsCount = {}
        self.traitsLikelyhood = {}
        self.classesCount = {}
        self.classesLikelyhood = {}
        self.alfa = 1
        self.current_dir = path.dirname(__file__)
        self.dataPath = self.current_dir
        
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
        self.relativePath = path.join('../', 'data',filename)
        # Absolute (!) path to the data file
        self.dataPath = path.abspath(self.relativePath)
        self.separator = separator
    
    def addClassValue(self: Self, value: str) -> None:
        """
        Adds a new class to classesCount or incements the amount of the existing class by 1
        """
        if value not in self.classesCount:
            self.classesCount[value] = 1
        else:
            self.classesCount[value] += 1

    def addTraitValues(self: Self, df: pd.DataFrame, row) -> None:
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
    
    def calculateTraitLikelyhoods(self: Self, df: pd.DataFrame) -> None:
        """
        Calculates likelyhoods of all values of all traits
        """
        for column in df.columns[1:]:
            # Each column has to have a value in this case, it will throw an error
            for key,value in self.traitsCount[column].items():
                if column not in self.traitsLikelyhood:
                    self.traitsLikelyhood[column] = {}
                self.traitsLikelyhood[column][key] = value/self.dataSize 
            
    def calculateClassesLikelyhoods(self: Self) -> None:
        """
        Calculates likelyhoods of each class of all classes
        """
        for key,value in self.classesCount.items():
            self.classesLikelyhood[key] = value/self.dataSize
            print(f"{value} / {self.dataSize}")

    def fit(self: Self) -> bool:
        """
        Trains the model based on the given data
        Arguments:
        Nothing
        Returns:
        True - if the data has been analyzed and trained successfully
        False - if the training set is not configured properly
        Raises Error - if an error occured while reading the data
        """
        # =============== Read data ===============
        
        # Read file
        if not self.dataPath:
            print("First set the data source!")
            return False
        try:
            df = pd.read_csv(self.dataPath)
        except Exception as error:
            print("An error has occured while reading the file (check if the file exists)", file=stderr)
            raise error
        
        self.dataSize = df.size//len(df.columns)
        # Set the class column name (usually 'class')
        classColumnName = df.columns[0]
        # Count classes and traits
        for index, row in df.iterrows():
            self.addClassValue(row[classColumnName])
            self.addTraitValues(df,row)

        # =============== Calculate likelyhoods ===============
        self.calculateTraitLikelyhoods(df)
        self.calculateClassesLikelyhoods()
        print(self.traitsLikelyhood)
        print(self.classesLikelyhood)

        # =============== Train model ===============
        # TODO

        # The model has been fit successfully
        return True
    
    def predict(self: Self) -> tuple[str,float]:
        """
        Predicts which value is most likely to appear
        Arguments:
        Nothing
        Returns:
         - A tuple with:
           > A string with class that has been chosen as the most likely
           > Probability of that class in the inverval [0,1]
        """
        # TODO
        pass

    def predict_proba(self: Self) -> List[float]:
        """
        Predicts which value is most likely to appear
        Arguments:
        Nothing
        Returns:
         - A list of probabilities of classification
        """
        # TODO
        pass