This application uses Tkinter, CV2, CNN, to analyze poker hands from images. It is designed to assist new players by evaluating their hands and providing feedback. The application offers two main features:

1. Hand Analysis
     
  Image Capture: Users can upload a photo or use a webcam to capture the hand they are dealt. Press 'C' to capture the image.
  Image Processing: Utilizes CV2 to find and crop the cards in the image. Each card is resized to 224x224 pixels to match the training specifications of the CNN.
  Card Classification: The images are analyzed by a custom Convolutional Neural Network (CNN) trained on 6240* images. Each card is identified and classified, e.g., "Jack of Diamonds."
  Hand Evaluation: The classified cards are written to a list as Card objects (with suit and rank attributes). The application evaluates the hand, identifying pairs (high, medium, low), high cards (Ace, King, Queen,   Jack), and categorizes the hand further based on kickers, suited cards, connected cards, and gapped cards.
  Data Handling: After processing, the card images are deleted from the system.

3. Community Card Analysis
  
  Image Upload: Users can upload a photo featuring the flop, turn, and river, along with two different hands.
  Image Segmentation: CV2 divides the image into three sections (top half, bottom half left, bottom half right) and processes each section individually.
  Card Processing: Similar to the hand analysis, cards are identified, cropped, resized, and classified.
  Comparative Analysis: The application uses an algorithm to compare the hands based on Texas Hold'em rules and determines the winner in relation to the community cards.
     
##there are two cnn building files in the repository, i tried using pytorch to build the cnn, but the model was overfitting and not giving the accuracy i wanted, so i moved to fastAI, which provided much better results, thought i should include both attempts, as both where functional


