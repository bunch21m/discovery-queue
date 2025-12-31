from flask import Flask, render_template, jsonify, request
from src.models.two_tower_candidate_pooler import TwoTowerRecommender
from src.db.user_functions import get_user_by_username
from src.db.interaction_functions import add_user_interaction_to_database

NUM_RECOMMENDATIONS = 10

# Initialize Two Tower Recommender
recommender_object = TwoTowerRecommender()

app = Flask(__name__, template_folder='frontend')

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

    recommended_games, recommended_app_ids, recommended_movie_urls = recommender_object.get_recommendations(username, k=NUM_RECOMMENDATIONS)
    try:
        recommendations = []
        # We iterate through a combined list of games, appIDs, and movie URLs,
        # adding it all to a list of dictionaries for easy consumption on the frontend.
        for game, app_id, movie_url in zip(recommended_games, recommended_app_ids, recommended_movie_urls):
            recommendations.append({
                "name": game,
                "appID": app_id,
                "movieURL": movie_url
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


# Run the application
if __name__ == '__main__':
    # 'debug=True' allows the server to reload automatically on code changes
    # Since this is running through docker, host = '0.0.0.0' is needed to be accessible from outside
    # the container
    app.run(debug=True, host='0.0.0.0', port=5000)
