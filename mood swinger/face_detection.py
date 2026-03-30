import cv2
import numpy as np
import logging
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize face detection cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Define expression states based on facial features
EXPRESSIONS = {
    'HAPPY': 'Smiling',
    'NEUTRAL': 'Neutral',
    'ALERT': 'Alert'
}

# Define colors for different expressions (BGR format)
EXPRESSION_COLORS = {
    'Smiling': (0, 255, 255),   # Yellow
    'Neutral': (0, 255, 0),     # Green
    'Alert': (0, 0, 255)        # Red
}

# Expression smoothing parameters
EXPRESSION_CHANGE_DELAY = 1.0  # Seconds between expression changes
MIN_DETECTION_COUNT = 3  # Number of consistent detections needed to change expression

# Quote display parameters
QUOTE_DISPLAY_DURATION = 5.0  # How long each quote stays on screen (seconds)
QUOTE_FADE_DURATION = 1.0    # Duration of fade in/out effect (seconds)

# Motivational quotes for each expression
MOTIVATIONAL_QUOTES = {
    'Smiling': [
        "Keep that smile going! It looks great on you!",
        "Your smile brightens everyone's day!",
        "Happiness is the best makeup one can wear.",
        "A smile is a curve that sets everything straight.",
        "Your positive energy is contagious!"
    ],
    'Neutral': [
        "Every day is a new opportunity to be amazing!",
        "Your potential is endless!",
        "Small progress is still progress.",
        "You are stronger than you know.",
        "Today is full of possibilities!"
    ],
    'Alert': [
        "Stay focused, you're doing great!",
        "Your attention to detail will pay off!",
        "Concentration is the secret of strength.",
        "Your dedication is inspiring!",
        "Success comes from staying sharp!"
    ]
}

import random

def get_motivational_quote(expression):
    """Get a random motivational quote based on the expression"""
    quotes = MOTIVATIONAL_QUOTES.get(expression, [])
    return random.choice(quotes) if quotes else "You're doing great!"

def get_emotion_color(emotion):
    # Define colors for different emotions (BGR format)
    colors = {
        'happy': (0, 255, 255),    # Yellow
        'sad': (255, 0, 0),        # Blue
        'angry': (0, 0, 255),      # Red
        'neutral': (0, 255, 0),    # Green
        'surprise': (255, 165, 0),  # Orange
        'fear': (128, 0, 128),     # Purple
        'disgust': (0, 128, 0)     # Dark Green
    }
    return colors.get(emotion.lower(), (0, 255, 0))  # Default to green if emotion not found

def get_expression(face_img):
    """Determine expression based on detected facial features"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes with more sensitive parameters
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,  # Reduced from 5
        minSize=(20, 20)  # Smaller minimum size
    )
    
    # Detect smile with more sensitive parameters
    smiles = smile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  # Reduced from 1.7
        minNeighbors=15,  # Reduced from 20
        minSize=(25, 25)
    )
    
    # More nuanced expression detection
    if len(smiles) > 0:
        return EXPRESSIONS['HAPPY']
    elif len(eyes) >= 2:
        return EXPRESSIONS['ALERT']
    else:
        return EXPRESSIONS['NEUTRAL']

def main():
    try:
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open webcam!")
            return
        
        logging.info("Webcam initialized successfully")
        logging.info("Expression detection started. Press 'q' to quit.")
        
        frame_count = 0
        last_expression = None
        last_expression_time = 0
        expression_counter = {}
        
        # Quote display variables
        current_quote = ""
        quote_start_time = 0
        last_quote_time = 0
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract the face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Determine current expression
                current_expression = get_expression(face_roi)
                
                # Update expression counter
                if current_expression not in expression_counter:
                    expression_counter = {current_expression: 1}
                else:
                    expression_counter[current_expression] += 1
                
                # Check if we should update the displayed expression
                if (current_time - last_expression_time >= EXPRESSION_CHANGE_DELAY and 
                    expression_counter.get(current_expression, 0) >= MIN_DETECTION_COUNT):
                    last_expression = current_expression
                    last_expression_time = current_time
                    expression_counter.clear()
                
                # Use the smoothed expression for display
                display_expression = last_expression if last_expression else current_expression
                color = EXPRESSION_COLORS[display_expression]
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Add expression label
                cv2.putText(display_frame, display_expression, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Handle quote updates and display
                if current_time - last_quote_time >= QUOTE_DISPLAY_DURATION:
                    if current_time - last_expression_time >= EXPRESSION_CHANGE_DELAY:
                        current_quote = get_motivational_quote(display_expression)
                        quote_start_time = current_time
                        last_quote_time = current_time
                
                # If we have a quote to display
                if current_quote:
                    # Calculate fade effect
                    quote_age = current_time - quote_start_time
                    fade_in = min(1.0, quote_age / QUOTE_FADE_DURATION)
                    fade_out = min(1.0, max(0, (QUOTE_DISPLAY_DURATION - quote_age) / QUOTE_FADE_DURATION))
                    alpha = min(fade_in, fade_out) * 0.7  # Max opacity 0.7
                    
                    # Create semi-transparent overlay for the quote
                    overlay = display_frame.copy()
                    
                    # Calculate quote position at the bottom of the frame
                    quote_y = display_frame.shape[0] - 40  # 40 pixels from bottom
                    
                    # Get the text size
                    text_size = cv2.getTextSize(current_quote, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    quote_x = (display_frame.shape[1] - text_size[0]) // 2  # Center horizontally
                    
                    # Draw dark background for better readability
                    cv2.rectangle(overlay, 
                                (0, quote_y - 30),
                                (display_frame.shape[1], quote_y + 10),
                                (0, 0, 0), -1)
                    
                    # Add the overlay with calculated transparency
                    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                    
                    # Add the quote text with calculated transparency
                    text_color = (int(255 * min(fade_in, fade_out)), 
                                int(255 * min(fade_in, fade_out)), 
                                int(255 * min(fade_in, fade_out)))
                    cv2.putText(display_frame, current_quote,
                              (quote_x, quote_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
                if frame_count % 30 == 0:  # Log every ~1 second
                    logging.info(f"Detected expression: {display_expression}")
            
            # Display the frame
            cv2.imshow('Expression Detection', display_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        logging.info("Program terminated by user")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise e
        
    finally:
        # Release resources
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        logging.info("Resources cleaned up")

if __name__ == "__main__":
    main()
