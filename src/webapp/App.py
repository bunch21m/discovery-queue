from src.models.Recommender import Recommender

NUM_RECOMMENDATIONS = 10

def main():
    recommenderFunction = Recommender.implementedRecommenders["simplePositiveRatingBased"]
    print(f"Recommended games: {recommenderFunction(numRecommendationsToMake=NUM_RECOMMENDATIONS)}")

if __name__ == "__main__":
    main()