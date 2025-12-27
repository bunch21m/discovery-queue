from src.models.Recommender import Recommender

NUM_RECOMMENDATIONS = 10

def main():
    recommenderObject = Recommender("simpleTagVectorBased", pathToDataset='data/games.json')
    recommendedGames, recommendedAppIDs = recommenderObject.getRecommendations(numRecommendationsToMake=NUM_RECOMMENDATIONS)
    
    while True:
        for i in range(len(recommendedGames)):
            print(f"{i + 1}. {recommendedGames[i]} (App ID: {recommendedAppIDs[i]})")
            userInput = input("Enter your interest level in this game (0-1), or 'q' to quit: ")
            if userInput.lower() == 'q':
                return
            try:
                interestLevel = float(userInput)
                if interestLevel < 0 or interestLevel > 1:
                    print("Interest level must be between 0 and 1.")
                    continue
                recommenderObject.recommenderObject.updateUserInterestVector(recommendedAppIDs[i], interestLevel)
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 1, or 'q' to quit.")
                continue

        recommendedGames, recommendedAppIDs = recommenderObject.getRecommendations(numRecommendationsToMake=NUM_RECOMMENDATIONS)

if __name__ == "__main__":
    main()
