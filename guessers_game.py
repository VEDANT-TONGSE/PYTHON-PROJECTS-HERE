import getpass  # To hide input for the secret number

def guessers_game():
    print("Welcome to the Guesser's Game!")
    print("Rules: One player (Chooser) selects a number, and other players (Guessers) try to guess it.")
    print("Let's get started!\n")
    
    # Step 1: Setup Players
    players = []

    num_players = int(input("Enter the number of players: "))
    for i in range(num_players):
        player_name = input(f"Enter the name of Player {i + 1}: ")
        players.append(player_name)
    
    # Step 2: Chooser selects a number
    print("\nChoose who will be the Chooser.")
    chooser = input("Enter the name of the Chooser: ")
    
    if chooser not in players:
        print("Invalid player name! Restart the game.")
        return

    print(f"\n{chooser}, you will now select a number for others to guess!")
    try:
        secret_number = int(getpass.getpass("Enter the secret number (it will be hidden): "))
    except ValueError:
        print("Invalid input! Please enter a valid number. Restart the game.")
        return
    
    # Step 3: Start Guessing Rounds
    max_attempts = 2  # Limit attempts to 5 rounds
    attempts = 0
    winners = []  # List to store the names of players who guessed correctly

    print("\nGuessers, try to guess the number! You have 5 attempts.\n")
    while attempts < max_attempts:
        guesses = []  # List to collect all players' guesses in the current round
        print(f"--- Round {attempts + 1} ---")
        
        for player in players:
            if player != chooser:
                try:
                    guess = int(input(f"{player}, enter your guess: "))
                except ValueError:
                    print("Invalid input! Enter a valid number.")
                    guess = None
                guesses.append((player, guess))
        
        # Check guesses and identify winners
        for player, guess in guesses:
            if guess == secret_number:
                winners.append(player)
        
        if winners:
            print(f"\nCongratulations! The following players guessed the number {secret_number} correctly: {', '.join(winners)}")
            return
        
        print("\nNo correct guesses in this round. Moving to the next round.")
        attempts += 1

    # Step 4: End Game - No winners, Chooser wins
    if not winners:
        print(f"\nGame Over! No one guessed the correct number. The secret number was {secret_number}.")
        print(f"The Chooser, {chooser}, wins!")
    else:
        print(f"\nCongratulations to the winners: {', '.join(winners)}")

    print("Thanks for playing!")

# Run the game
if __name__ == "__main__":
    guessers_game()
