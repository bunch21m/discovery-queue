from flask import Flask, render_template, jsonify, request
from src.models.recommender import Recommender
from src.db.user_functions import get_user_by_username
from src.db.interaction_functions import add_user_interaction_to_database
from src.db.tools.auto_wishlister import (
    action_for_score,
    score_app_id_for_user,
    set_user_config,
)

NUM_RECOMMENDATIONS = 10

# Initialize Two Tower Recommender
recommender_object = Recommender("twoTower")
# recommender_object = Recommender("river")

from src.ingest.train_two_tower_on_new_interactions import train_on_new_interactions

app = Flask(__name__, template_folder='frontend', static_folder='frontend', static_url_path='')

@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Endpoint to trigger incremental training on new interactions.
    """
    try:
        train_on_new_interactions()
        return jsonify({"message": "Incremental training complete!"})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({"error": str(e)}), 500

# Route for the home page (in this case, locahost:5000/ or 127.0.0.1:5000)
@app.route('/')
def home():
    """
    Renders the home page of the website.
    Loads from ./frontend/index.html
    """
    return render_template('index.html')

# Route for getting recommendations
# (On the frontend, this can be called in plain old javascript using something like fetch('/recommend'))
@app.route('/recommend')
def recommend():
    """
    Endpoint to get game recommendations.

    Returns:
            JSON: A JSON object containing recommended games and their appIDs.
    """
    # Get username from query params, default to 'test'
    username = request.args.get('username', 'test')

    recommended_games = recommender_object.get_recommendations(username, k=NUM_RECOMMENDATIONS)
    # print(f"Recommended games for user {username}: {recommended_games}")
    try:
        recommendations = []
        # We iterate through the recommended games and extract relevant information
        # from each game to send back to the frontend
        # The data field in each game contains additional information like movie URLs, screenshots, descriptions, etc.
        
        for game in recommended_games:
            movie_urls = []
            screenshot_urls = []
            genres = []
            categories = []
            short_description = ""
            detailed_description = ""
            required_age = ""
            price = ""
            # Number of positive reviews
            positive = ""
            # Number of negative reviews
            negative = ""
            if game:
                if 'movies' in game:
                    for movie in game['movies']:
                        movie_urls.append(movie)

                if 'screenshots' in game:
                    for screenshot in game['screenshots']:
                        screenshot_urls.append(screenshot)

                if 'genres' in game:
                    for genre in game['genres']:
                        genres.append(genre)

                if 'categories' in game:
                    for category in game['categories']:
                        categories.append(category)
                        
                short_description = game.get('short_description', "")

                detailed_description = game.get('detailed_description', "")

                required_age = game.get('required_age', "")

                price = game.get('price', "")

                positive = game.get('positive', "")

                negative = game.get('negative', "")
            recommendations.append({
                "name": game.get('name'),
                "appID": game.get('app_id'),
                "movieURLs": movie_urls,
                "screenshotURLs": screenshot_urls,
                "genres": genres,
                "categories": categories,
                "shortDescription": short_description,
                "detailedDescription": detailed_description,
                "requiredAge": required_age,
                "price": price,
                "positive": positive,
                "negative": negative
            })
        
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Route for recieving whether the user liked a recommendation or not
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    app_id = data.get('appID')
    feedback_type = data.get('feedback') # 'wishlist' or 'skip'
    username = data.get('username', 'test')

    if not app_id or not feedback_type:
        return jsonify({"error": "Missing appID or feedback"}), 400

    try:
        user_data = get_user_by_username(username)
        if not user_data:
            return jsonify({"error": f"User {username} not found"}), 404
        
        user_id = user_data['userid']
        
        success = add_user_interaction_to_database(app_id, user_id, feedback_type)
        
        if success:
            return jsonify({"message": f"Feedback {feedback_type} received"}), 200
        else:
            return jsonify({"error": "Failed to save interaction"}), 500

    except Exception as e:
        print(f"Error handling feedback: {e}")
        return jsonify({"error": str(e)}), 500
    

# Route for recieving information used to set up the auto wishlist/skip tool
@app.route('/auto_wishlist_config', methods=['POST'])
def auto_wishlist_setup():
    payload = request.get_json() or {}
    username = payload.get('username', 'test')
    config = payload.get('config', {})

    stored_config = set_user_config(username, config)

    return jsonify({
        "message": "Configuration saved",
        "config": stored_config
    }), 200


# Route for recieving information on whether the auto wishlist/skip tool would
# wishlist or skip the game with the given appID
@app.route('/auto_wishlist_decision', methods=['POST'])
def auto_wishlist_decision():
    payload = request.get_json() or {}
    app_id = payload.get('appID') or payload.get('appId')
    username = payload.get('username', 'test')

    if not app_id:
        return jsonify({"error": "Missing appID"}), 400

    try:
        score = score_app_id_for_user(app_id, username)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    action = action_for_score(score)
    return jsonify({
        "appID": app_id,
        "score": score,
        "action": action
    }), 200

# Run the application
if __name__ == '__main__':
    # 'debug=True' allows the server to reload automatically on code changes
    # Since this is running through docker, host = '0.0.0.0' is needed to be accessible from outside
    # the container
    app.run(debug=True, host='0.0.0.0', port=5000)
