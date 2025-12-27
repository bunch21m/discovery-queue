from src.models.CommonModelUtils import loadAllGamesFromJSON

class SimpleTagVectorBasedRecommender:
    """
    A simple recommender that recommends games based on tag vectors.
    It loads a dataset of games from a JSON file, computes similarity based on tags,
    and recommends the top numRecommendationsToMake similar games.

    When initializing, it loads the dataset of games from the specified JSON file.
    Every game in the dataset is expected to have a 'tags' field which is a list of tags.
    Otherwise it is discarded from the dataset.

    Every tag encountered is added to a master list of tags.
    Each game's tags are then converted into a tag vector, which is itself then normalized.
    Every game is then represented by its normalized tag vector.

    Note that the 2 norm (Euclidean norm) = sum_i sqrt(x_i^2) is used for normalization.

    The class will also store a vector representation of the user's interests in tags.
    Every time the updateUserInterestVector function is called with a game and an interest level,
    the user's interest vector is updated by adding the game's tag vector multiplied by the interest level,
    with interest levels ranging from 0 to 1. The user's interest vector is also normalized after each update.

    When getRecommendations is called, the recommender computes the cosine similarity between the user's interest vector
    and each game's tag vector, and recommends the top numRecommendationsToMake games with the highest similarity scores.

    We use cosine similarity rather than some other measure like Euclidean distance because cosine similarity
    focuses on the orientation of the vectors rather than their magnitude, which is more appropriate for our
    use case of recommending games based on user interests.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes a new SimpleTagVectorBasedRecommender.
        Loads the dataset of games from a JSON file.

        Args:
            **kwargs: Has a key "pathToDataset" which is the path to the dataset file.
        """
        for key, value in kwargs.items():
            if key == "pathToDataset":
                pathToDataset = value

        self.dataset = loadAllGamesFromJSON(pathToDataset)
        self.tagList = set()

        # First pass: build the master tag list
        for app in self.dataset:
            game = self.dataset[app]

            tags = game['tags']
            for tag in tags:
                self.tagList.add(tag)

        self.tagList = list(self.tagList)
        self.tagIndexMap = {tag: idx for idx, tag in enumerate(self.tagList)}
        self.gameTagVectors = {}

        # Second pass: build each game's tag vector
        for app in self.dataset:
            game = self.dataset[app]
            tags = game['tags']

            tagVector = [0] * len(self.tagList)
            for tag in tags:
                if tag in self.tagIndexMap:
                    tagVector[self.tagIndexMap[tag]] += 1

            # We use L2 normalization for the tag vectors
            norm = sum(x**2 for x in tagVector) ** 0.5
            if norm > 0:
                tagVector = [x / norm for x in tagVector]

            self.gameTagVectors[app] = tagVector

        # Initialize user interest vector to zero vector
        self.userInterestVector = [0] * len(self.tagList)

    def updateUserInterestVector(self, gameAppID: str, interestLevel: float):
        """
        Updates the user's interest vector based on their interest level in a game.
        This also removes a game from consideration for future recommendations.

        Args:
            gameAppID (str): The App ID of the game.
            interestLevel (float): The user's interest level in the game, ranging from 0 to 1.
        """
        if gameAppID not in self.gameTagVectors:
            return

        gameTagVector = self.gameTagVectors[gameAppID]

        # Update user interest vector
        for i in range(len(self.tagList)):
            self.userInterestVector[i] += gameTagVector[i] * interestLevel

        # Normalize user interest vector
        norm = sum(x**2 for x in self.userInterestVector) ** 0.5
        if norm > 0:
            self.userInterestVector = [x / norm for x in self.userInterestVector]

        # Remove the game from consideration for future recommendations
        if gameAppID in self.dataset:
            del self.dataset[gameAppID]
            del self.gameTagVectors[gameAppID]

    def getRecommendations(self, **kwargs):
        """
        Recommends games based on the user's interest vector.

        Args:
            **kwargs: Has a key "numRecommendationsToMake" which is the number of recommendations to make.

        Returns:
            list: A list of recommended game names.
            list: A list of recommended game App IDs.
        """
        for key, value in kwargs.items():
            if key == "numRecommendationsToMake":
                numRecommendationsToMake = value

        # Base case: if user interest vector is zero, we will simply default to top rated games
        # based on positive ratings, just like the SimplePositiveRatingBasedRecommender
        if all(x == 0 for x in self.userInterestVector):
            sortedGames = sorted(self.dataset.values(), key=lambda x: x['positive'], reverse=True)
            recommendations = [game['name'] for game in sortedGames[:numRecommendationsToMake]]
            appIDs = [game['appID'] for game in sortedGames[:numRecommendationsToMake]]
            return recommendations, appIDs
        
        # Otherwise, we must compute cosine similarity between user interest vector and each game's tag vector
        # for every game in the dataset
        similarityScores = []
        for app in self.dataset:
            game = self.dataset[app]
            gameTagVector = self.gameTagVectors[app]

            # Compute cosine similarity
            dotProduct = sum(self.userInterestVector[i] * gameTagVector[i] for i in range(len(self.tagList)))
            normGame = sum(x**2 for x in gameTagVector) ** 0.5
            normUser = sum(x**2 for x in self.userInterestVector) ** 0.5

            if normGame > 0 and normUser > 0:
                cosineSimilarity = dotProduct / (normGame * normUser)
            else:
                cosineSimilarity = 0

            similarityScores.append((cosineSimilarity, game['name'], app))

        # Sort games by similarity score in descending order
        similarityScores.sort(key=lambda x: x[0], reverse=True)
        recommendations = [similarityScores[i][1] for i in range(numRecommendationsToMake)]
        appIDs = [similarityScores[i][2] for i in range(numRecommendationsToMake)]

        return recommendations, appIDs