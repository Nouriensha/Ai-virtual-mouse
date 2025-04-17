import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
draw_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

holding = False
pinch_start_time = None
action_label = ""

# Timing control
last_left_click_time = 0
last_right_click_time = 0
click_cooldown = 0.5  # seconds

# Helper functions
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def is_folded(base_y, tip_y):
    return tip_y > base_y

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)

    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            draw_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            fingers_tip_ids = [4, 8, 12, 16, 20]
            fingers_base_ids = [2, 5, 9, 13, 17]

            fingers_status = []
            for tip_id, base_id in zip(fingers_tip_ids, fingers_base_ids):
                tip = landmarks[tip_id]
                base = landmarks[base_id]
                folded = is_folded(base.y, tip.y)
                fingers_status.append(not folded)  # True = Open, False = Folded

            all_fingers_open = all(fingers_status)
            all_fingers_folded = not any(fingers_status)
            index_open = fingers_status[1]
            middle_open = fingers_status[2]
            thumb_open = fingers_status[0]

            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            thumb_tip = landmarks[4]

            index_x = int(index_tip.x * frame_width)
            index_y = int(index_tip.y * frame_height)
            middle_x = int(middle_tip.x * frame_width)
            middle_y = int(middle_tip.y * frame_height)
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            # Calculate distances
            thumb_middle_dist = distance(thumb_x, thumb_y, middle_x, middle_y)

            current_time = time.time()

            # Pinch Detection (Thumb + Middle Finger)
            pinch_threshold = 35  # tighter distance
            pinch_detected = thumb_middle_dist < pinch_threshold

            if pinch_detected:
                if not holding:
                    if pinch_start_time is None:
                        pinch_start_time = current_time
                    elif (current_time - pinch_start_time) > 0.2:  # 200ms stable pinch
                        pyautogui.mouseDown()
                        holding = True
                        action_label = "Hold & Drag Started"
                else:
                    # Now move the mouse WHILE holding (dragging)
                    cursor_x = int((thumb_x + middle_x) / 2)
                    cursor_y = int((thumb_y + middle_y) / 2)
                    screen_x = int(screen_width / frame_width * cursor_x)
                    screen_y = int(screen_height / frame_height * cursor_y)
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
                    action_label = "Dragging"
            else:
                pinch_start_time = None
                if holding:
                    pyautogui.mouseUp()
                    holding = False
                    action_label = "Released (End Drag)"

                # Normal operation (move/click/scroll)
                if all_fingers_open:
                    cursor_x = int((index_x + middle_x) / 2)
                    cursor_y = int((index_y + middle_y) / 2)
                    screen_x = int(screen_width / frame_width * cursor_x)
                    screen_y = int(screen_height / frame_height * cursor_y)
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
                    action_label = "Moving"

                elif all_fingers_folded:
                    pyautogui.scroll(-30)
                    action_label = "Scrolling Down"

                elif thumb_open and not index_open and not middle_open:
                    pyautogui.scroll(30)
                    action_label = "Scrolling Up"

                elif (not index_open) and middle_open:
                    if (current_time - last_left_click_time) > click_cooldown:
                        pyautogui.click(button='left')
                        last_left_click_time = current_time
                        action_label = "Left Click"

                elif (not middle_open) and index_open:
                    if (current_time - last_right_click_time) > click_cooldown:
                        pyautogui.click(button='right')
                        last_right_click_time = current_time
                        action_label = "Right Click"

                else:
                    action_label = "Idle"
