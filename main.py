import models
import tools
import tensorflow as tf
import gc
import numpy
import json
from pluginFactory import PluginFactory
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from enum import Enum

# Constants
MODEL_SUBJECTS_PATH = "data/modelSubjects.tflearn"
MODEL_TYPES_PATH = "data/modelTypes.tflearn"
MODEL_VALUES_PATH = "data/modelValues.tflearn"

class ModelType(Enum):
    SUBJECTS = 1
    TYPES = 2
    VALUES = 3

class ModelLoader(ABC):
    @abstractmethod
    def load_model(self) -> models.Model:
        pass

class ModelLoaderImpl(ModelLoader):
    def __init__(self, path: str, dictionnary: Dict, subjects_or_types: List, model_type: ModelType):
        self.path = path
        self.dictionnary = dictionnary
        self.subjects_or_types = subjects_or_types
        self.model_type = model_type
        self.model = None

    def load_model(self) -> models.Model:
        try:
            if self.model_type == ModelType.SUBJECTS:
                self.model = models.getModelSubjects(self.dictionnary, self.subjects_or_types)
            elif self.model_type == ModelType.TYPES:
                self.model = models.getModelTypes(self.dictionnary, self.subjects_or_types)
            elif self.model_type == ModelType.VALUES:
                self.model = models.getModelValues(self.dictionnary)
            self.model.load(self.path)
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

class SentenceAnalyser:
    def __init__(self, model_subjects: models.Model, model_types: models.Model, model_values: models.Model):
        self.model_subjects = model_subjects
        self.model_types = model_types
        self.model_values = model_values

    def _predict(self, model: models.Model, sentence: str, dictionnary: Dict, stopwords: List) -> Tuple:
        try:
            return model.predict([tools.bagOfWords(sentence, dictionnary, stopwords)])[0]
        except Exception as e:
            print(f"Error predicting sentence: {e}")
            return None

    def analyse(self, sentence: str, subjects: List, types: List, stopwords: List, dictionnary: Dict) -> Tuple:
        try:
            predicted_subject = self._predict(self.model_subjects, sentence, dictionnary, stopwords)
            predicted_type = self._predict(self.model_types, sentence, dictionnary, stopwords)
            predicted_value = self._predict(self.model_values, sentence, dictionnary, stopwords)
            return predicted_subject, predicted_type, predicted_value
        except Exception as e:
            print(f"Error analysing sentence: {e}")
            return None, None, None

class AnswerSearcher:
    def __init__(self):
        pass

    def search_answer(self, sentence: str, subject, type_s) -> str:
        plugin = PluginFactory.getPlugin(subject, type_s)
        return plugin.response(sentence)

class SentenceProcessor:
    def __init__(self, sentence_analyser: SentenceAnalyser, answer_searcher: AnswerSearcher):
        self.sentence_analyser = sentence_analyser
        self.answer_searcher = answer_searcher

    def process_sentence(self, sentence: str, subjects: List, types: List, stopwords: List, dictionnary: Dict) -> str:
        result_subject, result_type, result_value = self.sentence_analyser.analyse(sentence, subjects, types, stopwords, dictionnary)
        answer = self.answer_searcher.search_answer(sentence, result_subject, result_type)
        return answer

def main():
    subjects, types, stopwords, dictionnary = tools.defaultValues()

    model_subjects_loader = ModelLoaderImpl(MODEL_SUBJECTS_PATH, dictionnary, subjects, ModelType.SUBJECTS)
    model_types_loader = ModelLoaderImpl(MODEL_TYPES_PATH, dictionnary, types, ModelType.TYPES)
    model_values_loader = ModelLoaderImpl(MODEL_VALUES_PATH, dictionnary, None, ModelType.VALUES)

    model_subjects = model_subjects_loader.load_model()
    model_types = model_types_loader.load_model()
    model_values = model_values_loader.load_model()

    sentence_analyser = SentenceAnalyser(model_subjects, model_types, model_values)
    answer_searcher = AnswerSearcher()
    sentence_processor = SentenceProcessor(sentence_analyser, answer_searcher)

    while True:
        print("Tape your sentence:")
        test = input()
        answer = sentence_processor.process_sentence(test, subjects, types, stopwords, dictionnary)
        print(f"Answer: {answer}")

if __name__ == '__main__':
    main()
