from src.models.CommonModelUtils import loadAllGamesFromDatabase

class SimplePositiveRatingBasedRecommender:
    """
    A simple recommender that recommends games based on their positive ratings.
    It loads a dataset of games from the database, sorts them by positive ratings,
    and recommends the top numRecommendationsToMake games.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes a new SimplePositiveRatingBasedRecommender.
        Loads the dataset of games from the database, then sorts them by positive ratings.
        """
        self.dataset = loadAllGamesFromDatabase()
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
        numRecommendationsToMake = 10 # Default
        for key, value in kwargs.items():
            if key == "numRecommendationsToMake":
                numRecommendationsToMake = value

        bestGames = self.dataset[:numRecommendationsToMake]
        appIDs = [game['appID'] for game in bestGames]
        recommendations = [game['name'] for game in bestGames]

        return recommendations, appIDs
