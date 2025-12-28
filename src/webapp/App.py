from flask import Flask, render_template, jsonify
from src.models.Recommender import Recommender

NUM_RECOMMENDATIONS = 10

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
    recommenderObject = Recommender("simplePositiveRatingBased")
    recommendedGames, recommendedAppIDs = recommenderObject.getRecommendations(numRecommendationsToMake=NUM_RECOMMENDATIONS)
    try:
        recommendations = []
        # We iterate through a combined list of games and appIDs to create the response
        # adding both the name and appID for each recommended game to the response list as
        # a dictionary. In the end, we return a JSON version of a list of these dictionaries.
        for game, appID in zip(recommendedGames, recommendedAppIDs):
            recommendations.append({
                "name": game,
                "appID": appID
            })
        
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Run the application
if __name__ == '__main__':
    # 'debug=True' allows the server to reload automatically on code changes
    # Since this is running through docker, host = '0.0.0.0' is needed to be accessible from outside
    # the container
    app.run(debug=True, host='0.0.0.0', port=5000)
