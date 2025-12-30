from flask import Flask, render_template, jsonify, request
from src.models.recommender import Recommender

NUM_RECOMMENDATIONS = 10

recommender_object = Recommender("simpleTagVectorBased")

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
    recommended_games, recommended_app_ids, recommended_movie_urls = recommender_object.get_recommendations(num_recommendations_to_make=NUM_RECOMMENDATIONS)
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
def get_user_feedback():
    """
    Endpoint to receive user feedback on recommendations.

    Expects a JSON payload with 'appID' and 'feedback' fields.
    'feedback' should be either 0, 1, 2, 3, or 4 representing:
        0 - Strongly Dislike
        1 - Dislike
        2 - Neutral
        3 - Like
        4 - Strongly Like

    Returns:
            JSON: A JSON object confirming receipt of feedback.
    """
    data = request.get_json()
    app_id = data.get('appID')
    feedback = data.get('feedback')
    # print(f"Received feedback for App ID {app_id}: {feedback}")
    # return jsonify({"status": "Feedback received"}), 200
    # Maps feedback levels to floats for the tag vector recommender
    feedback_mapping = {
        0: 0.0,  # Strongly Dislike
        1: 0.25, # Dislike
        2: 0.5,  # Neutral
        3: 0.75, # Like
        4: 1.0   # Strongly Like
    }
    interest_level = feedback_mapping.get(feedback, 0.5)  # Default to Neutral if invalid feedback
    recommender_object.recommender_object.update_user_interest_vector(app_id, interest_level)
    print(f"Updated user interest vector with App ID {app_id} and interest level {interest_level}")
    return jsonify({"status": "Feedback received"}), 200


# Run the application
if __name__ == '__main__':
    # 'debug=True' allows the server to reload automatically on code changes
    # Since this is running through docker, host = '0.0.0.0' is needed to be accessible from outside
    # the container
    app.run(debug=True, host='0.0.0.0', port=5000)
