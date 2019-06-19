"""
The purpose of this library is to solve the code challenge at:
https://agileengine.bitbucket.io/beKIvpUlPMtzhfAy/
"""

import os
import sys
import logging
import typing
import math

from lxml import etree as ET

from difflib import SequenceMatcher
from functools import reduce

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def sigmoid(x):
    """
    Sigmoid function.
    """
    return 1 / (1 + math.exp(-x))


class Text(object):
    """
    Text entity.
    This class provides text analyzers.
    """

    def __init__(self, text: str) -> None:
        """
        @param text: Any text string.
        """
        if not text:
            text = ""
        if not isinstance(text, str):
            raise TypeError(type(text), text)
        self.__text = text

    def __str__(self) -> str:
        """
        String serializer.
        """
        return "<Text: '{}'.".format(self.get_title())

    def get_raw_text(self) -> str:
        """
        String getter.
        """
        return self.__text

    def get_title(self) -> str:
        """
        Pretty string serializer.
        """
        return self.__text.strip()

    def get_normalized_text(self) -> str:
        """
        This function normalizes a text.
        It can be tokenized, lemmatized, etc, but not in this demo.
        """
        return ''.join(
            character
            for character in self.__text.lower()
            if character.isalnum()
        )

    def __eq__(self, other: object) -> bool:
        """
        Returns True if elements are equal.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(other)
        return self.get_normalized_text() == other.get_normalized_text()

    def get_match_score(self, other: object) -> float:
        """
        Compares 2 Texts and returns a score
        between 0 and 1.
        """
        logger.debug("Comparing text %s and %s.", self, other)
        if not isinstance(other, self.__class__):
            raise TypeError(other)
        score = SequenceMatcher(None,
                                self.get_normalized_text(),
                                other.get_normalized_text()).ratio()
        logger.debug("Score for %s and %s is: %s", self, other, score)
        return score


class Element(object):
    """
    XML Element entity.
    """

    def __init__(self, element: ET.Element, xpath: str) -> None:
        """
        @param element: Any XML element.
        """
        if not isinstance(element, ET._Element):
            raise TypeError(element)
        self.__element = element
        if not isinstance(xpath, str):
            raise TypeError(xpath)
        self.__xpath = xpath

    def get_xpath(self) -> str:
        """
        XPath getter. 
        """
        return self.__xpath

    def __str__(self) -> str:
        """
        Element serializer.
        """
        return "<Element: '{}' '{}'.".format(self.tag,
                                             self.text.get_title())

    @property
    def text(self) -> str:
        """
        Element text getter.
        """
        return Text(self.__element.text)

    @property
    def tag(self) -> str:
        """
        Element tag getter.
        """
        return self.__element.tag

    @property
    def element_id(self) -> str:
        """
        Returns the element ID, if exists.
        """
        element_id = self.attributes.get("id")
        if element_id:
            return element_id.get_raw_text()
        return None

    @property
    def attributes(self) -> dict:
        """
        Element attributes getter.
        """
        return {
            k: Text(v)
            for k, v in self.__element.attrib.items()
            if v
        }

    def get_match_score(self, other: object) -> float:
        """
        Compares 2 Elements and returns a score
        between 0 and 1.
        """
        logger.debug("Comparing elements %s and %s.", self, other)
        if not isinstance(other, self.__class__):
            raise TypeError(other)
        score = self.text.get_match_score(other.text)
        attributes_scores = (
            v1.get_match_score(v2)
            for k1, v1 in self.attributes.items()
            for k2, v2 in other.attributes.items()
            if k1 == k2
        )
        attributes_scores = [
            score
            for score in attributes_scores
        ]
        score += sum(attributes_scores)
        if self.tag != other.tag:
            score /= 2
        score = sigmoid(score)
        assert score >= 0, score
        assert score <= 1, score
        logger.debug("Score for %s and %s is: %s", self, other, score)
        return score


class Dataset(object):
    """
    Dataset file entity.
    """

    def __init__(self, path: str) -> None:
        """
        @param path: Dataset file path.
        """
        logger.debug("Initializing dataset: %s.", path)
        self.__path = path
        if not os.path.isfile(self.__path):
            raise OSError("Invalid dataset path:", self.__path)
        self.__file_buffer = None
        self.__tree = ET.parse(self.__path)

    def get_element_by_id(self, element_id: str) -> Element:
        """
        Find element by ID.
        """
        logger.debug("Searching for: %s.", element_id)
        xpath = "//*[@id='{}']".format(element_id)
        results = self.__tree.xpath(xpath)
        logger.debug("Found elements %s.", results)
        if not results:
            raise RuntimeError("No element found with ID:", element_id)
        element = results[0]
        xpath = self.__tree.getelementpath(element)
        logger.debug("Found: %s.", element)
        return Element(element, xpath)

    def get_all_elements(self) -> typing.Generator:
        """
        File reader.
        """
        logger.debug("Reading dataset: %s.", self)
        for element in self.__tree.iter():
            try:
                xpath = self.__tree.getelementpath(element)
            except ValueError:
                xpath = ''
            yield Element(element, xpath)
        logger.debug("Finished reading lines from: %s.", self)

    def __str__(self) -> str:
        """
        String serializer.
        """
        return "<Dataset: '{}'>".format(self.__path)


class PredictionModel(object):
    """
    Prediction model entity.
    """

    DEFAULT_THRESHOLD = 0.5
    DEFAULT_LIMIT = 3

    def __init__(self, target: str) -> None:
        """
        @param target: Target ID or text string.
        """
        if not target:
            raise ValueError(target)
        if not isinstance(target, str):
            raise TypeError(target)
        self.__target = Text(target)
        self.__target_element = None

    def __str__(self) -> str:
        """
        String serializer.
        """
        return "<PredictionModel: '{}'>".format(self.__target)

    def is_target_learned(self) -> bool:
        """
        Returns True if target has been found
        in the training dataset.
        """
        return self.__target_element is not None

    def learn(self, dataset: Dataset) -> None:
        """
        Learning from Dataset.
        """
        logger.debug("Learning from: %s.", dataset)
        if not isinstance(dataset, Dataset):
            raise TypeError(dataset)
        self.__target_element = dataset.get_element_by_id(self.__target.get_raw_text())
        logger.debug("Finished learning from: %s.", dataset)

    def find(self, dataset: Dataset,
             limit: int=DEFAULT_LIMIT,
             threshold: float=DEFAULT_THRESHOLD) -> None:
        """
        Finding target in dataset.
        """
        logger.debug("Searching target in: %s.", dataset)
        if not isinstance(dataset, Dataset):
            raise TypeError(dataset)
        if not self.is_target_learned():
            raise AttributeError("Not trained.")
        if not isinstance(threshold, (int, float)):
            raise TypeError(threshold)
        if threshold > 1:
            raise ValueError(threshold)
        if threshold < 0:
            raise ValueError(threshold)
        if not isinstance(limit, int):
            raise TypeError(int)
        if limit <= 0:
            raise ValueError(limit)
        scores = (
            (
                element,
                self.__target_element.get_match_score(element)
            )
            for element in dataset.get_all_elements()
        )
        scores = (
            (
                element,
                score
            )
            for element, score in scores
            if score > threshold
        )
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        logger.debug("Elements above %s found: %s.", threshold, scores)
        return scores[:limit]


class Report(object):
    """
    Report entity.
    """

    SEPARATOR = "-" * 30
    PAD = " " * 4

    def __init__(self, model: PredictionModel) -> None:
        """
        @param model: PredictionModel instance.
        """
        self.__model = model
        self.__results = {}

    def add(self, title: str, matches: list) -> None:
        """
        Add result to report.
        """
        logger.debug("Adding report for '%s': %s.", title, len(matches))
        if not title:
            raise ValueError(title)
        if not isinstance(title, str):
            raise TypeError(title)
        if not isinstance(matches, list):
            raise TypeError(matches)
        if title in self.__results:
            raise RuntimeError("Already added.")
        self.__results[title] = matches

    def show(self, stdout=sys.stdout) -> str:
        """
        Print report to STDOUT.
        """
        if self.__results:
            for title, matches in self.__results.items():
                for match in matches:
                    print("'{}' (score={})".format(match[0].get_xpath(),
                                                   match[1]), file=stdout)


if __name__ == "__main__":
    logger.debug("[Init]")
    if len(sys.argv) < 2:
        raise RuntimeError("Missing source dataset.")
    if len(sys.argv) < 3:
        raise RuntimeError("Missing target dataset.")
    if os.path.isfile(sys.argv[-1]):
        target_id = "make-everything-ok-button"
        variants = sys.argv[2:]
    else:
        target_id = sys.argv[-1]
        variants = sys.argv[2:-1]
    results = []
    try:
        model = PredictionModel(target_id)
        report = Report(model=model)
        source = Dataset(sys.argv[1])
        model.learn(source)
        for path in variants:
            dataset = Dataset(path)
            matches = model.find(dataset, limit=1, threshold=0.8)
            report.add(title=path, matches=matches)
    except:
        logger.exception("[Error]")
        raise
    else:
        logger.debug("[OK]")
        report.show()
    finally:
        logger.debug("[End]")
