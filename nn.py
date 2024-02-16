import cv2
from PIL import Image
import os
from fastai.vision.all import load_learner
from fastai.vision.all import*
import tkinter as tk
from tkinter import messagebox, filedialog
import time
from itertools import combinations
from collections import Counter

########################variables########################
totalCardsList = []
communityCardsList = []
leftHandList = []
rightHandList = []
learner = load_learner('models\card_classifier_update.pkl')
categories=('Ace of clubs','Ace of diamonds','ace of hearts','ace of spades','eight of clubs','eight of diamonds','eight of hearts','eight of spades','five of clubs','five of diamonds','five of hearts', 'five of spades','four of clubs','four of diamonds','four of hearts','four of spades','jack of clubs','jack of diamonds','jack of hearts','jack of spades','joker','king of clubs','king of diamonds','king of hearts','king of spades','nine of clubs','nine of diamonds','nine of hearts','nine of spades','queen of clubs','queen of diamonds','queen of hearts','queen of spades','seven of clubs','seven of diamonds','seven of hearts','seven of spades','six of clubs','six of diamonds','six of hearts','six of spades','ten of clubs','ten of diamonds','ten of hearts','ten of spades','three of clubs','three of diamonds','three of hearts',' three of spades',' two of clubs', 'two of diamonds',' two of hearts','two of spades')



########################class########################

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    def __str__(self):
        return f"{self.number} {self.suit}"
    
    def get_value(self):
        value_order = ['Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace']
        return value_order.index(self.rank) + 2
    
    
########################methods########################

def classify_image(img):
    img = img.resize((224, 224))
    pred, idx, probs = learner.predict(img)
    return pred


def ClassSearch():
    global totalCardsList
    folder_path = './liveCards/YourHand'
    file_list = os.listdir(folder_path)
    totalCardsSet = set()
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        frame = cv2.imread(file_path)
        cropped_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped_img = Image.fromarray(cropped_img)
        card_classification = classify_image(cropped_img)
        print(card_classification)
        rank, suit = card_classification.split(" of ")
        new_card = Card(rank, suit)
        totalCardsSet.add(new_card)

    totalCardsList = list(totalCardsSet)
    if len(totalCardsList) == 2:
        GoodHandAlgorithm()
    else:
        messagebox.showerror("Error", "Invalid number of cards. You should have exactly two cards.")


def GoodHandAlgorithm():
    root = tk.Tk()
    root.withdraw()

    if len(totalCardsList) != 2:
        messagebox.showerror("Error", "Invalid number of cards. You should have exactly two cards in Texas Hold'em pre-flop.")
        root.destroy()
        return
    card1, card2 = totalCardsList
    ranks = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "jack": 11, "queen": 12, "king": 13, "ace": 14}

    # Check if the two cards form a pair
    has_pair = card1.rank.lower() == card2.rank.lower()

    # Check if either of the two cards is a high card (Ace, King, Queen, or Jack)
    has_high_card = card1.rank.lower() in ["ace", "king", "queen", "jack"] or card2.rank.lower() in ["ace", "king", "queen", "jack"]

    # Check if both cards are of the same suit, which might lead to a flush
    is_suited = card1.suit == card2.suit

    # Calculate the difference in ranks to check for connectedness or gaps
    # Connected cards are one rank apart, while gap cards are 2 to 3 ranks apart
    rank_diff = abs(ranks[card1.rank.lower()] - ranks[card2.rank.lower()])
    is_connected = rank_diff == 1  # True if cards are consecutive
    has_gap = 1 < rank_diff <= 3  # True if there is a small gap between the cards
    message = ""


    if has_pair:
        pair_rank = card1.rank.lower()
        message += f"Pair of {pair_rank}'s. "
        if pair_rank in ["ace", "king", "queen", "jack"]:
            message += "This is a strong hand. Consider playing aggressively, especially in later positions. High pairs often warrant raising pre-flop.\n"
        elif pair_rank in ["ten", "nine", "eight", "seven"]:
            message += "A medium pair can be a good hand but play cautiously. Consider the position and actions of other players before raising.\n"
        else: # 2-6
            message += "Low pairs are vulnerable and should be played cautiously. Consider calling to see the flop but be ready to fold against aggressive raises.\n"
    elif has_high_card:
        message += "Having a high card can be advantageous, but much depends on your kicker (the second card). With a strong kicker, consider playing more aggressively. Be cautious with weak kickers.\n"
    else:
        message += "Without high cards or a pair, your hand is weak. Consider folding unless you can see the flop cheaply. Be prepared to fold if the flop doesn't improve your hand.\n"

    if is_suited:
        message += "Suited cards increase your chances of making a flush. This adds value to your hand, especially if the cards are also high or connected.\n"
    if is_connected:
        message += "Connected cards have a good chance of forming a straight. The closer they are, the better your odds.\n"
    if has_gap:
        message += f"Your cards have a gap of {rank_diff}. This reduces your straight potential, but it's still possible. Play such hands cautiously.\n"
    os.remove('image.jpg')
    file_list = os.listdir('liveCards\YourHand')
    for file_name in file_list:
        file_path = os.path.join('liveCards\YourHand', file_name)
        os.remove(file_path)
    messagebox.showinfo("Hand Analysis", message)
    root.destroy()

def ProcessAndAppenendToList(path, card_list):
    file_list = os.listdir(path)
    card_set = set()

    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        frame = cv2.imread(file_path)
        cropped_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped_img = Image.fromarray(cropped_img)
        card_classification = classify_image(cropped_img)
        print(card_classification)
        rank, suit = card_classification.split(" of ")
        new_card = Card(rank, suit)
        card_set.add(new_card)
        os.remove(file_path)

    if card_list == "community":
        communityCardsList.extend(list(card_set))
    elif card_list == "left":
        leftHandList.extend(list(card_set))
    elif card_list == "right":
        rightHandList.extend(list(card_set))

def UploadImageAndSubdivide():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image = Image.open(file_path)
        image.save("image.jpg")
        print("Image saved")
    image = cv2.imread('image.jpg')
    if image is None:
        raise ValueError("The image could not be loaded.")
    height, width, _ = image.shape
    top_height = height // 2
    left_width = width // 2
    image1 = image[:top_height, :]  # Top half
    image2 = image[top_height:, :left_width]  # Bottom left quarter
    image3 = image[top_height:, left_width:]  # Bottom right quarter
    cv2.imwrite('1.jpg', image1)
    cv2.imwrite('2.jpg', image2)
    cv2.imwrite('3.jpg', image3)

    print("Images have been saved.")
def CropCards(image_path, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    image = cv2.imread(image_path)
    display_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1000: 
            x, y, w, h = cv2.boundingRect(contour)
            card = image[y:y+h, x:x+w]
            file_path = os.path.join(save_directory, f'card{idx+1}.jpg')
            try:
                resizedCard = cv2.resize(card, (224, 224)) 
                cv2.imwrite(file_path, resizedCard)
                print(f"card{idx+1}: {file_path}")
            except Exception as e:
                print(f"Error saving image: {e}")


def GetImagesIntoPath():
    UploadImageAndSubdivide()
    CropCards("1.jpg", "./liveCards/ComCards")
    CropCards("2.jpg", "./liveCards/LeftHand")
    CropCards("3.jpg", "./liveCards/RightHand")
    os.remove('1.jpg')
    os.remove('2.jpg')
    os.remove('3.jpg')
    print("Images cropped")
    
def WhoWonAlgorithm():
    global communityCardsList, leftHandList, rightHandList
    communityCardsList.clear()
    leftHandList.clear()
    rightHandList.clear()
    GetImagesIntoPath()
    com_path = './liveCards/ComCards'
    left_path = './liveCards/LeftHand'
    right_path = './liveCards/RightHand'
    ProcessAndAppenendToList(com_path, "community")
    ProcessAndAppenendToList(left_path, "left")
    ProcessAndAppenendToList(right_path, "right")
    left_hands = all_hands(leftHandList, communityCardsList)
    right_hands = all_hands(rightHandList, communityCardsList)

    best_left_hand = best_hand(left_hands)
    best_right_hand = best_hand(right_hands)

    left_hand_rank = hand_rank(best_left_hand)
    right_hand_rank = hand_rank(best_right_hand)

    if left_hand_rank > right_hand_rank:
        winner = "Player on the left"
        winning_hand_description = describe_hand(left_hand_rank)
    else:
        winner = "Player on the right"
        winning_hand_description = describe_hand(right_hand_rank)

    messagebox.showinfo("Hand Analysis", f"Winner: {winner} wins with a {winning_hand_description}")
    os.remove('image.jpg')


def hand_rank(hand):
    # Mapping values to ranks
    value_map = {'Two': 2, 'two': 2, 'Three': 3, 'three':3, 'Four': 4,'four':4, 'Five': 5, 'five':5, 'Six': 6, 'six':6, 'Seven': 7, 'seven':7, 'Eight': 8,'eight':8, 'Nine': 9,'nine': 9, 'Ten': 10, 'ten': 10, 'Jack': 11, 'jack': 11, 'Queen': 12, 'queen':12, 'King': 13, 'king': 13, 'Ace': 14, 'ace':14}
    ranks = sorted([value_map[c.rank] for c in hand], reverse=True)
    suits = [c.suit for c in hand]
    is_flush = len(set(suits)) == 1

    # Check for a straight, considering Ace as both high and low
    is_straight = False
    ranks_set = set(ranks)
    if len(ranks_set) == 5:  # Ensure all ranks are unique
        # Check for standard straight
        if ranks[0] - ranks[4] == 4:
            is_straight = True
        # Check for Ace-low straight (A-2-3-4-5)
        if ranks_set == {14, 2, 3, 4, 5}:
            is_straight = True
            ranks = [5, 4, 3, 2, 1]  # Adjust ranks for Ace-low straight

    rank_counts = Counter(ranks)
    most_common = rank_counts.most_common(2)
    highest_count, highest_rank = most_common[0]
    second_highest_count, _ = most_common[1] if len(most_common) > 1 else (0, None)

    if is_straight and is_flush:
        return (8, ranks)
    if highest_count == 4:
        return (7, highest_rank, ranks)
    if highest_count == 3 and second_highest_count == 2:
        return (6, highest_rank, ranks)
    if is_flush:
        return (5, ranks)
    if is_straight:
        return (4, ranks)
    if highest_count == 3:
        return (3, highest_rank, ranks)
    if highest_count == 2 and second_highest_count == 2:
        return (2, ranks)
    if highest_count == 2:
        return (1, highest_rank, ranks)
    return (0, ranks)

def best_hand(hands):
    return max(hands, key=hand_rank)

def all_hands(player_hand, community_cards):
    return [list(combo) for combo in combinations(player_hand + community_cards, 5)]

def describe_hand(hand_rank_value):
    hand_types = [
        "High Card", "Pair", "Two Pairs", "Three of a Kind",
        "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
    ]
    return hand_types[hand_rank_value[0]]


################camera################
def getCards():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image = Image.open(file_path)
        image.save("image.jpg")
        print("Image saved")
    image = cv2.imread('image.jpg')
    if image is None:
        raise ValueError("The image could not be loaded.")
    if not os.path.exists('liveCards\YourHand'):
        os.makedirs('liveCards\YourHand')
    image = cv2.imread('./image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1000: 
            x, y, w, h = cv2.boundingRect(contour)
            card = image[y:y+h, x:x+w]
            file_path = os.path.join('liveCards\YourHand', f'card{idx+1}.jpg')
            try:
                resizedCard = cv2.resize(card, (224, 224)) 
                cv2.imwrite(file_path, resizedCard)
                print(f"card{idx+1}: {file_path}")
            except Exception as e:
                print(f"Error saving image: {e}")
    ClassSearch()
        

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Card Detector")
    root.geometry("300x300")
    label = tk.Label(root, text="Poker", font=("Arial", 16))
    label.pack(pady=10)  # Add some vertical padding
    who_won_button = tk.Button(root, text="Who won?", command=WhoWonAlgorithm) # selected = false
    who_won_button.pack(pady=5)  # Add some vertical padding
    good_hand_button = tk.Button(root, text="Good hand?", command=getCards) # selected = true	
    good_hand_button.pack(pady=5)  # Add some vertical padding
    root.mainloop()
    

