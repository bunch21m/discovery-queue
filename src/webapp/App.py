from src.models.Recommender import Recommender

NUM_RECOMMENDATIONS = 10

def main():
    # Set up the five different interest levels for a user's interest in a game
    gameMap = {}
    gameMap[0] = [] # Strong dislike
    gameMap[1] = [] # Slight dislike
    gameMap[2] = [] # Indifferent
    gameMap[3] = [] # Slight like
    gameMap[4] = [] # Strong like

    recommenderObject = Recommender("simplePositiveRatingBased", pathToDataset='data/games.json')
    recommendedGames = recommenderObject.getRecommendations(numRecommendationsToMake=NUM_RECOMMENDATIONS)
    for game in recommendedGames:
        print(game)

if __name__ == "__main__":
    main()