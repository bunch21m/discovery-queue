from river import compose
from river import linear_model
from river import preprocessing
from src.db.tools.game_functions import load_all_games_from_database
from src.db.user_functions import get_user_by_username
from src.db.interaction_functions import get_users_interactions_from_database

class RiverRecommender:
    """
    A simple online learning recommender using the river library.
    It trains a Logistic Regression model on the user's past interactions
    (wishlist vs skip) using game features like positive/negative ratings and price.
    """

    def __init__(self):
        self.games_dataset = load_all_games_from_database()

    def get_recommendations(self, username, k=10):
        """
        Gets game recommendations for the given user.

        Args:
            username (str): The username of the user to get recommendations for.
            k (int): The number of recommendations to return.

        Returns:
            list: A list of recommended games.
        """

        user = get_user_by_username(username)
        if not user:
            return []
        
        user_id = user['userid']
        interactions = get_users_interactions_from_database(user_id)
        
        # Initialize model
        # We use a standard scaler to normalize features and a logistic regression for binary classification
        model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )
        
        interacted_app_ids = set()
        
        # ============ Training Phase ============
        # Currently, when feedback is recieved in app.py, it is just added to the database,
        # and not sent to the model directly. Thus, we must either find that single new interaction
        # and train on it, or retrain from scratch on all interactions.  Here, we retrain from scratch.
        # It is almost certainly better to find the new interaction only and train on that,
        # but this is simpler to implement for now.

        # interactions is a dict {interaction_id: row}
        for interaction in interactions.values():
            app_id = str(interaction['appid'])
            interacted_app_ids.add(app_id)
            
            if app_id in self.games_dataset:
                game = self.games_dataset[app_id]
                x = {
                    'positive': game['positive'],
                    'negative': game['negative'],
                    'price': game['price']
                }
                
                # Label: True if wishlist, False if skip
                y = interaction['interactiontype'] == 'wishlist'
                
                model.learn_one(x, y)


        # ============ Recommendation Phase ============   
        # Predict for candidates (games not yet interacted with)
        candidates = []
        for app_id, game in self.games_dataset.items():
            if app_id not in interacted_app_ids:
                x = {
                    'positive': game['positive'],
                    'negative': game['negative'],
                    'price': game['price']
                }
                # predict_proba_one returns a dict {label: probability}
                # We want the probability of 'True' (wishlist)
                score = model.predict_proba_one(x).get(True, 0.0)
                candidates.append((score, game))
                
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k games
        return [c[1] for c in candidates[:k]]
