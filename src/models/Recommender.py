from src.models.SimplePositiveRatingBasedRecommender import SimplePositiveRatingBasedRecommender
from src.models.SimpleTagVectorBasedRecommender import SimpleTagVectorBasedRecommender

class Recommender:
    """
    Represents a recommendation system that
    recommends steam games. Has the centralized list of all implemented recommenders.

    All recommenders must implement a function getRecommendations which takes in 
    some various arguments and returns a list of recommended games.
    """

    # Mapping from recommender name to the class that provides recommendations
    # The class must implement a getRecommendations method
    implementedRecommenders = {
        "simplePositiveRatingBased": SimplePositiveRatingBasedRecommender,
        "simpleTagVectorBased": SimpleTagVectorBasedRecommender
    }

    def __init__(self, recommenderType: str, *args, **kwargs) -> None:
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
        self.recommenderObject = Recommender.implementedRecommenders[recommenderType](*args, **kwargs)

    def getRecommendations(self, *args, **kwargs):
        """
        Gets recommendations from the underlying recommender object.

        Args:
            **kwargs: Arguments to pass to the recommender's getRecommendations method.

        Returns:
            list: A list of recommended games, and their appIDs.
        """
        return self.recommenderObject.getRecommendations(*args, **kwargs)
