import os
import json

def getRecommendations(*args, **kwargs):
    """
    Recommends a game simply by returning the top 
    numRecommendationsToMake highest rated games.
    """
    for key, value in kwargs.items():
        if key == "numRecommendationsToMake":
            numRecommendationsToMake = value

    dataset = {}
    if os.path.exists('data/games.json'):
        with open('data/games.json', 'r', encoding='utf-8') as fin:
            text = fin.read()
            if len(text) > 0:
                dataset = json.loads(text)

    for app in dataset:
        game = dataset[app]                         

        name = game['name']         # Game name (string).
        positive = game['positive'] # Positive votes (int).
        
    # Sort games by positive ratings in descending order
    sortedGames = sorted(dataset.values(), key=lambda x: x['positive'], reverse=True)
    
    sortedGames = sortedGames[:numRecommendationsToMake]
    sortedGames = [game['name'] for game in sortedGames]

    return sortedGames