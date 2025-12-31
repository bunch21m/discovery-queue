from src.models.simple_positive_rating_based_recommender import SimplePositiveRatingBasedRecommender
from src.models.simple_tag_vector_based_recommender import SimpleTagVectorBasedRecommender
from src.models.two_tower_candidate_pooler import TwoTowerRecommender

class Recommender:
    """
    Represents a recommendation system that
    recommends steam games. Has the centralized list of all implemented recommenders.

    All recommenders must implement a function get_recommendations which takes in 
    some various arguments and returns a list of recommended games.
    """

    # Mapping from recommender name to the class that provides recommendations
    # The class must implement a get_recommendations method
    implemented_recommenders = {
        "simplePositiveRatingBased": SimplePositiveRatingBasedRecommender,
        "simpleTagVectorBased": SimpleTagVectorBasedRecommender,
        "twoTower": TwoTowerRecommender
    }

    def __init__(self, recommender_type: str, *args, **kwargs) -> None:
        """
        Initializes a new recommender with the given recommender type.

        Args:
            recommender_type (str): The type of recommender logic to use.
        
        Raises:
            ValueError: If the recommender type is not supported.
        """
        if recommender_type not in Recommender.implemented_recommenders:
            raise ValueError(f"Unknown recommender type '{recommender_type}'.")
        
        self.recommender_type = recommender_type
        self.recommender_object = Recommender.implemented_recommenders[recommender_type](*args, **kwargs)

    def get_recommendations(self, *args, **kwargs):
        """
        Gets recommendations from the underlying recommender object.

        Args:
            **kwargs: Arguments to pass to the recommender's get_recommendations method.

        Returns:
            list: A list of recommended games, and their appIDs.
        """
        return self.recommender_object.get_recommendations(*args, **kwargs)
