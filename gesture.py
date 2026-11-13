import math
import time
import math
import cv2
import mediapipe as mp
import pyautogui
from signal import signal,SIGINT

def handler(signal_received, frame):
    print('SIGINT or CTRL-C detected. Exiting gracefully.')
    exit(0)

if __name__ == '__main__':   
    signal(SIGINT,handler)    

    MODEL_PATH = "hand_landmarker.task"

    # Gesture configuration
    PINCH_START_THRESHOLD = 0.045
    PINCH_RELEASE_THRESHOLD = 0.070

    DRAG_HOLD_SECONDS = 0.35
    CLOSE_HOLD_SECONDS = 1.0
    CLOSE_COOLDOWN_SECONDS = 2.0

    CURSOR_SMOOTHING = 0.25
    SCROLL_SENSITIVITY = 1500
    SCROLL_DEAD_ZONE = 0.012


    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode


    def distance(point1, point2):
        return math.hypot(
            point1.x - point2.x,
            point1.y - point2.y,
        )


    def finger_is_extended(landmarks, tip_id, pip_id):
        """
        With an upright hand, the fingertip is above the PIP joint.
        In image coordinates, a smaller Y value means higher.
        """
        return landmarks[tip_id].y < landmarks[pip_id].y


    def four_fingers_folded(landmarks):
        """
        Detect a fist using index, middle, ring and little fingers.
        The thumb is intentionally ignored because its orientation
        changes significantly between left and right hands.
        """
        return all(
            landmarks[tip_id].y > landmarks[pip_id].y
            for tip_id, pip_id in (
                (8, 6),    # Index
                (12, 10),  # Middle
                (16, 14),  # Ring
                (20, 18),  # Little finger
            )
        )


    def draw_landmarks(frame, landmarks):
        frame_height, frame_width, _ = frame.shape

        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

        for connection in connections:
            start = landmarks[connection.start]
            end = landmarks[connection.end]

            start_point = (
                int(start.x * frame_width),
                int(start.y * frame_height),
            )
            end_point = (
                int(end.x * frame_width),
                int(end.y * frame_height),
            )

            cv2.line(
                frame,
                start_point,
                end_point,
                (100, 100, 100),
                2,
            )

        for landmark in landmarks:
            point = (
                int(landmark.x * frame_width),
                int(landmark.y * frame_height),
            )

            cv2.circle(
                frame,
                point,
                4,
                (0, 255, 0),
                -1,
            )


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")


    screen_width, screen_height = pyautogui.size()

    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True


    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )


    # Cursor smoothing state
    smooth_x = screen_width / 2
    smooth_y = screen_height / 2

    # Pinch, click and drag state
    pinch_active = False
    pinch_start_time = 0.0
    pinch_start_x = 0
    pinch_start_y = 0
    dragging = False

    # Scroll state
    scroll_active = False
    previous_scroll_y = None

    # Close-window gesture state
    fist_start_time = None
    last_close_time = 0.0


    try:
        with HandLandmarker.create_from_options(options) as hand_landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)

                frame_height, frame_width, _ = frame.shape

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = time.monotonic_ns() // 1_000_000

                result = hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                current_time = time.monotonic()
                current_gesture = "No hand"

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]

                    draw_landmarks(frame, landmarks)

                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    middle_tip = landmarks[12]

                    index_extended = finger_is_extended(
                        landmarks,
                        8,
                        6,
                    )
                    middle_extended = finger_is_extended(
                        landmarks,
                        12,
                        10,
                    )
                    ring_extended = finger_is_extended(
                        landmarks,
                        16,
                        14,
                    )
                    little_extended = finger_is_extended(
                        landmarks,
                        20,
                        18,
                    )

                    fist_detected = four_fingers_folded(landmarks)

                    pinch_distance = distance(
                        thumb_tip,
                        index_tip,
                    )

                    # Hysteresis: use different thresholds for starting
                    # and releasing the pinch.
                    if pinch_active:
                        pinching = (
                            pinch_distance < PINCH_RELEASE_THRESHOLD
                        )
                    else:
                        pinching = (
                            pinch_distance < PINCH_START_THRESHOLD
                        )

                    # Scroll gesture:
                    # index and middle extended, ring and little folded.
                    scroll_gesture = (
                        index_extended
                        and middle_extended
                        and not ring_extended
                        and not little_extended
                        and not pinching
                    )

                    # -------------------------------------------------
                    # CLOSE ACTIVE WINDOW: fist held for one second
                    # -------------------------------------------------

                    if fist_detected and not dragging:
                        current_gesture = "Hold fist to close"

                        if fist_start_time is None:
                            fist_start_time = current_time

                        fist_duration = current_time - fist_start_time

                        progress = min(
                            fist_duration / CLOSE_HOLD_SECONDS,
                            1.0,
                        )

                        bar_width = 250
                        filled_width = int(bar_width * progress)

                        cv2.rectangle(
                            frame,
                            (10, 75),
                            (10 + bar_width, 100),
                            (80, 80, 80),
                            2,
                        )

                        cv2.rectangle(
                            frame,
                            (10, 75),
                            (10 + filled_width, 100),
                            (0, 0, 255),
                            -1,
                        )

                        if (
                            fist_duration >= CLOSE_HOLD_SECONDS
                            and current_time - last_close_time
                            >= CLOSE_COOLDOWN_SECONDS
                        ):
                            # Windows: close the active window.
                            pyautogui.hotkey("alt", "f4")

                            last_close_time = current_time
                            fist_start_time = None

                    else:
                        fist_start_time = None

                    # -------------------------------------------------
                    # SCROLL: index and middle fingers raised
                    # -------------------------------------------------
                    if scroll_gesture and not dragging:
                        current_gesture = "Scroll"

                        current_scroll_y = (
                            index_tip.y + middle_tip.y
                        ) / 2

                        if not scroll_active:
                            scroll_active = True
                            previous_scroll_y = current_scroll_y

                        else:
                            delta_y = previous_scroll_y - current_scroll_y

                            if abs(delta_y) > 0.015:

                                scroll_amount = int(delta_y * 1000)

                                pyautogui.scroll(scroll_amount)

                                # Reset reference only after an actual scroll
                                previous_scroll_y = current_scroll_y

                    # -------------------------------------------------
                    # CURSOR, CLICK AND DRAG
                    # -------------------------------------------------

                    if (
                        not fist_detected
                        and not scroll_gesture
                        and index_extended
                    ):
                        target_x = int(index_tip.x * screen_width)
                        target_y = int(index_tip.y * screen_height)

                        target_x = max(
                            0,
                            min(screen_width - 1, target_x),
                        )
                        target_y = max(
                            0,
                            min(screen_height - 1, target_y),
                        )

                        # Exponential cursor smoothing.
                        smooth_x += (
                            target_x - smooth_x
                        ) * CURSOR_SMOOTHING

                        smooth_y += (
                            target_y - smooth_y
                        ) * CURSOR_SMOOTHING

                        pyautogui.moveTo(
                            int(smooth_x),
                            int(smooth_y),
                        )

                        if pinching:
                            if not pinch_active:
                                pinch_active = True
                                pinch_start_time = current_time

                                # Save where the pinch started.
                                pinch_start_x = int(smooth_x)
                                pinch_start_y = int(smooth_y)

                            pinch_duration = (
                                current_time - pinch_start_time
                            )

                            if (
                                pinch_duration >= DRAG_HOLD_SECONDS
                                and not dragging
                            ):
                                # Start dragging from the original
                                # pinch location.
                                pyautogui.moveTo(
                                    pinch_start_x,
                                    pinch_start_y,
                                )
                                pyautogui.mouseDown()

                                dragging = True

                            if dragging:
                                current_gesture = "Dragging"
                            else:
                                current_gesture = "Pinch: release to click"

                        elif pinch_active:
                            # The pinch has just been released.
                            if dragging:
                                pyautogui.mouseUp()
                                dragging = False
                                current_gesture = "Drag released"

                            else:
                                # Short pinch: normal click.
                                pyautogui.click(
                                    pinch_start_x,
                                    pinch_start_y,
                                )
                                current_gesture = "Click"

                            pinch_active = False

                        elif not dragging:
                            current_gesture = "Move"

                    elif pinch_active and not pinching:
                        # Safety release if the hand changes gesture.
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False

                        pinch_active = False

                    # Highlight thumb and index tips.
                    index_point = (
                        int(index_tip.x * frame_width),
                        int(index_tip.y * frame_height),
                    )
                    thumb_point = (
                        int(thumb_tip.x * frame_width),
                        int(thumb_tip.y * frame_height),
                    )

                    cv2.circle(
                        frame,
                        index_point,
                        10,
                        (0, 255, 255),
                        -1,
                    )

                    cv2.circle(
                        frame,
                        thumb_point,
                        10,
                        (0, 255, 255),
                        -1,
                    )

                    cv2.line(
                        frame,
                        index_point,
                        thumb_point,
                        (255, 0, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Pinch: {pinch_distance:.3f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                else:
                    # Release the mouse if tracking is lost while dragging.
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

                    pinch_active = False
                    scroll_active = False
                    previous_scroll_y = None
                    fist_start_time = None

                cv2.putText(
                    frame,
                    current_gesture,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow("Virtual Mouse", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        # Avoid leaving the mouse button pressed after an error.
        if dragging:
            pyautogui.mouseUp()

        cap.release()
        cv2.destroyAllWindows()