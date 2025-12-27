# Holds common utility functions and constants for models

import os
import json


def loadAllGamesFromJSON(pathToDataset: str):
    """
    Loads all games from a JSON dataset file.

    Args:
            pathToDataset (str): The path to the JSON dataset file.
    Returns:
            dict: A dictionary of all games loaded from the dataset holding many features.
    """
    dataset = {}
    if os.path.exists(pathToDataset):
        with open(pathToDataset, 'r', encoding='utf-8') as fin:
            text = fin.read()
            if len(text) > 0:
                dataset = json.loads(text)

    for appID in dataset:
        dataset[appID]['appID'] = appID
    
    return dataset
