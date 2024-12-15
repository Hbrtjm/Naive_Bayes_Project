import pandas as pd
import numpy as np
from typing import Self, List

class NaiveBayes:
    def __init__(self: Self) -> None:
        self.likelyHoods = []
        self.traits = []
        self.alfa = 1
    def fit(self: Self):
        pass
    def predict(self: Self) -> float:
        pass
    def predict_proba(self: Self) -> List[float]:
        pass