from src.models.SimplePositiveRatingBasedRecommender import getRecommendations as getPositiveRecommendations

class Recommender:
    """
    Represents a recommendation system that
    recommends steam games. Has the centralized list of all implemented recommenders.

    All recommenders must implement a function which takes in some various arguments
    and returns a list of exactly NUM_RECOMMENDATIONS recommended games.
    """

    # Mapping from recommender name to the function that provides recommendations
    implementedRecommenders = {
        "simplePositiveRatingBased": getPositiveRecommendations
    }

    def __init__(self, recommenderType: str) -> None:
        """
        Initializes a new recommender with the given recommender type.

        Args:
            recommenderType (str): The type of recommender logic to use.
        
        Raises:
            ValueError: If the recommender type is not supported.
        """
        if recommenderType not in Recommender.implementedRecommenders:
            raise ValueError(f"Unknown recommender type '{recommenderType}'.")
        self.recommenderType = recommenderType
