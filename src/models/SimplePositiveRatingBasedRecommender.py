import os
import json

class SimplePositiveRatingBasedRecommender:

    def __init__(self, **kwargs) -> None:
        """
        Initializes a new SimplePositiveRatingBasedRecommender.
        Loads the dataset of games from a JSON file, then sorts them by positive ratings.

        Args:
            **kwargs: Has a key "pathToDataset" which is the path to the dataset file.
        """
        for key, value in kwargs.items():
            if key == "pathToDataset":
                pathToDataset = value

        self.dataset = {}
        if os.path.exists(pathToDataset):
            with open(pathToDataset, 'r', encoding='utf-8') as fin:
                text = fin.read()
                if len(text) > 0:
                    self.dataset = json.loads(text)

        for app in self.dataset:
            game = self.dataset[app]

            name = game['name']         # Game name (string).
            positive = game['positive'] # Positive votes (int).

        self.dataset = sorted(self.dataset.values(), key=lambda x: x['positive'], reverse=True)

    def getRecommendations(self, **kwargs):
        """
        Recommends a game simply by returning the top 
        numRecommendationsToMake highest rated games.
        """
        for key, value in kwargs.items():
            if key == "numRecommendationsToMake":
                numRecommendationsToMake = value

        recommendations = self.dataset[:numRecommendationsToMake]
        recommendations = [game['name'] for game in recommendations]

        return recommendations
        