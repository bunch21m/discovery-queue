from src.models.common_model_utils import load_all_games_from_database

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
        self.dataset = load_all_games_from_database()
        self.dataset = sorted(self.dataset.values(), key=lambda x: x['positive'], reverse=True)

    def get_recommendations(self, **kwargs):
        """
        Recommends a game simply by returning the top 
        num_recommendations_to_make highest rated games.

        Args:
            **kwargs: Has a key "num_recommendations_to_make" which is the number of recommendations to make.

        Returns:
            list: A list of recommended game names.
            list: A list of recommended game App IDs.
            list: A list of movie URLs.
        """
        num_recommendations_to_make = 10 # Default
        for key, value in kwargs.items():
            if key == "num_recommendations_to_make":
                num_recommendations_to_make = value

        best_games = self.dataset[:num_recommendations_to_make]
        app_ids = [game['app_id'] for game in best_games]
        # We either return no movie if the game has no associated movie URLs,
        # or the first in its list of movie URLs for simplicity.
        movie_urls = []
        for game in best_games:
            movies = game.get('movies', [])
            if movies and isinstance(movies, list):
                first_movie = movies[0]
                movie_urls.append(first_movie)
            else:
                movie_urls.append(None)
        recommendations = [game['name'] for game in best_games]

        return recommendations, app_ids, movie_urls
