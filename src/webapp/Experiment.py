from src.db.interactionFunctions import addUserInteractionToDatabase, getUsersInteractionsFromDatabase
from src.db.userFunctions import addUserToDatabase, getUserByUsername, getAllUsersFromDatabase

def main():
    print(getAllUsersFromDatabase())
    
    print("Adding user...")
    addUserToDatabase("user2", {})
    print("Getting user by username...")
    user = getUserByUsername("user2")
    print(user)
    
    print("Adding interaction...")
    addUserInteractionToDatabase("578080", user['userid'], "skip")
    
    print("Getting user interactions...")
    interactions = getUsersInteractionsFromDatabase(user['userid'])
    print(interactions)
    
    print("Done.")

if __name__ == "__main__":
    main()