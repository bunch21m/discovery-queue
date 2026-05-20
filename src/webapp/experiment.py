from src.db.interaction_functions import add_user_interaction_to_database, get_users_interactions_from_database
from src.db.user_functions import add_user_to_database, get_user_by_username, get_all_users_from_database
from src.models.user_two_tower_embedding import concat_user_features

def main():
    print(get_all_users_from_database())
    
    print("Adding user...")
    add_user_to_database("user2", {})
    print("Getting user by username...")
    user = get_user_by_username("user2")
    print(user)
    
    print("Adding interaction...")
    add_user_interaction_to_database("578080", user['userid'], "wishlist")
    
    print("Getting user interactions...")
    interactions = get_users_interactions_from_database(user['userid'])
    print(interactions)
    
    print(concat_user_features("user2"))
    
    print("Done.")

if __name__ == "__main__":
    main()