from src.db.tools.game_functions import load_all_games_from_database

class SimpleTagVectorBasedRecommender:
    """
    A simple recommender that recommends games based on tag vectors.
    It loads a dataset of games from the database, computes similarity based on tags,
    and recommends the top num_recommendations_to_make similar games.

    When initializing, it loads the dataset of games from the database.
    Every game in the dataset is expected to have a 'tags' field which is a list of tags.
    Otherwise it is discarded from the dataset.

    Every tag encountered is added to a master list of tags.
    Each game's tags are then converted into a tag vector, which is itself then normalized.
    Every game is then represented by its normalized tag vector.

    Note that the 2 norm (Euclidean norm) = sum_i sqrt(x_i^2) is used for normalization.

    The class will also store a vector representation of the user's interests in tags.
    Every time the update_user_interest_vector function is called with a game and an interest level,
    the user's interest vector is updated by adding the game's tag vector multiplied by the interest level,
    with interest levels ranging from 0 to 1. The user's interest vector is also normalized after each update.

    When get_recommendations is called, the recommender computes the cosine similarity between the user's interest vector
    and each game's tag vector, and recommends the top num_recommendations_to_make games with the highest similarity scores.

    We use cosine similarity rather than some other measure like Euclidean distance because cosine similarity
    focuses on the orientation of the vectors rather than their magnitude, which is more appropriate for our
    use case of recommending games based on user interests.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes a new SimpleTagVectorBasedRecommender.
        Loads the dataset of games from the database.
        """
        self.dataset = load_all_games_from_database()
        self.tag_list = set()

        # First pass: build the master tag list
        for app in self.dataset:
            game = self.dataset[app]

            tags = game['tags']
            for tag in tags:
                self.tag_list.add(tag)

        self.tag_list = list(self.tag_list)
        self.tag_index_map = {tag: idx for idx, tag in enumerate(self.tag_list)}
        self.game_tag_vectors = {}

        # Second pass: build each game's tag vector
        for app in self.dataset:
            game = self.dataset[app]
            tags = game['tags']

            tag_vector = [0] * len(self.tag_list)
            for tag in tags:
                if tag in self.tag_index_map:
                    tag_vector[self.tag_index_map[tag]] += 1

            # We use L2 normalization for the tag vectors
            norm = sum(x**2 for x in tag_vector) ** 0.5
            if norm > 0:
                tag_vector = [x / norm for x in tag_vector]

            self.game_tag_vectors[app] = tag_vector

        # Initialize user interest vector to zero vector
        self.user_interest_vector = [0] * len(self.tag_list)

    def update_user_interest_vector(self, game_app_id: str, interest_level: float):
        """
        Updates the user's interest vector based on their interest level in a game.
        This also removes a game from consideration for future recommendations.

        Args:
            game_app_id (str): The App ID of the game.
            interest_level (float): The user's interest level in the game, ranging from 0 to 1.
        """
        if game_app_id not in self.game_tag_vectors:
            return

        game_tag_vector = self.game_tag_vectors[game_app_id]

        # Update user interest vector
        for i in range(len(self.tag_list)):
            self.user_interest_vector[i] += game_tag_vector[i] * interest_level

        # Normalize user interest vector
        norm = sum(x**2 for x in self.user_interest_vector) ** 0.5
        if norm > 0:
            self.user_interest_vector = [x / norm for x in self.user_interest_vector]

        # Remove the game from consideration for future recommendations
        if game_app_id in self.dataset:
            del self.dataset[game_app_id]
            del self.game_tag_vectors[game_app_id]

    def get_recommendations(self, **kwargs):
        """
        Recommends games based on the user's interest vector.

        Args:
            **kwargs: Has a key "num_recommendations_to_make" which is the number of recommendations to make.

        Returns:
            list: A list of recommended game names.
            list: A list of recommended game App IDs.
        """
        num_recommendations_to_make = 10  # Default
        for key, value in kwargs.items():
            if key == "num_recommendations_to_make":
                num_recommendations_to_make = value

        # Base case: if user interest vector is zero, we will simply default to top rated games
        # based on positive ratings, just like the SimplePositiveRatingBasedRecommender
        if all(x == 0 for x in self.user_interest_vector):
            sorted_games = sorted(self.dataset.values(), key=lambda x: x['positive'], reverse=True)
            recommendations = [game['name'] for game in sorted_games[:num_recommendations_to_make]]
            app_ids = [game['app_id'] for game in sorted_games[:num_recommendations_to_make]]
            movie_urls = []
            for game in sorted_games[:num_recommendations_to_make]:
                movies = game.get('movies', [])
                if movies and isinstance(movies, list):
                    first_movie = movies[0]
                    movie_urls.append(first_movie)
                else:
                    movie_urls.append(None)

            return recommendations, app_ids, movie_urls
        
        # Otherwise, we must compute cosine similarity between user interest vector and each game's tag vector
        # for every game in the dataset
        similarity_scores = []
        for app in self.dataset:
            game = self.dataset[app]
            movies = game.get('movies', [])
            first_movie = None
            if movies and isinstance(movies, list):
                first_movie = movies[0]
            game_tag_vector = self.game_tag_vectors[app]

            # Compute cosine similarity
            dot_product = sum(self.user_interest_vector[i] * game_tag_vector[i] for i in range(len(self.tag_list)))
            norm_game = sum(x**2 for x in game_tag_vector) ** 0.5
            norm_user = sum(x**2 for x in self.user_interest_vector) ** 0.5

            if norm_game > 0 and norm_user > 0:
                cosine_similarity = dot_product / (norm_game * norm_user)
            else:
                cosine_similarity = 0

            similarity_scores.append((cosine_similarity, game['name'], app, first_movie))

        # Sort games by similarity score in descending order
        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        recommendations = [similarity_scores[i][1] for i in range(num_recommendations_to_make)]
        app_ids = [similarity_scores[i][2] for i in range(num_recommendations_to_make)]
        movie_urls = [similarity_scores[i][3] for i in range(num_recommendations_to_make)]

        return recommendations, app_ids, movie_urls
