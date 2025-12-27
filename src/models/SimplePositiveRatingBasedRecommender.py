from src.models.CommonModelUtils import loadAllGamesFromJSON

class SimplePositiveRatingBasedRecommender:
    """
    A simple recommender that recommends games based on their positive ratings.
    It loads a dataset of games from a JSON file, sorts them by positive ratings,
    and recommends the top numRecommendationsToMake games.
    """

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

        self.dataset = loadAllGamesFromJSON(pathToDataset)
        self.dataset = sorted(self.dataset.values(), key=lambda x: x['positive'], reverse=True)

    def getRecommendations(self, **kwargs):
        """
        Recommends a game simply by returning the top 
        numRecommendationsToMake highest rated games.

        Args:
            **kwargs: Has a key "numRecommendationsToMake" which is the number of recommendations to make.

        Returns:
            list: A list of recommended game names.
            list: A list of recommended game App IDs.
        """
        for key, value in kwargs.items():
            if key == "numRecommendationsToMake":
                numRecommendationsToMake = value

        bestGames = self.dataset[:numRecommendationsToMake]
        appIDs = [game['appID'] for game in bestGames]
        recommendations = [game['name'] for game in bestGames]

        return recommendations, appIDs
        