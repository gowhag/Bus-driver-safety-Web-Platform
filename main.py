import cv2
import time
import mediapipe as mp
import simpleaudio as sa

# ----------------------------
ALERT_TIME = 5  # seconds eyes must be closed to trigger alarm
ALARM_FILE = "alarm.wav"  # put this in the same folder as the script
# ----------------------------

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Eye landmarks from Mediapipe Face Mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    # Convert normalized landmarks to pixel coords
    coords = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_indices]
    # EAR formula: vertical / horizontal
    A = ((coords[1][0]-coords[5][0])**2 + (coords[1][1]-coords[5][1])**2)**0.5
    B = ((coords[2][0]-coords[4][0])**2 + (coords[2][1]-coords[4][1])**2)**0.5
    C = ((coords[0][0]-coords[3][0])**2 + (coords[0][1]-coords[3][1])**2)**0.5
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for eye closed detection (tweak if needed)
EAR_THRESHOLD = 0.25

# Initialize camera
cap = cv2.VideoCapture(0)
closed_eyes_start = None
alarm_playing = None

def play_alarm():
    global alarm_playing
    if alarm_playing is None or not alarm_playing.is_playing():
        wave_obj = sa.WaveObject.from_wave_file(ALARM_FILE)
        alarm_playing = wave_obj.play()

print("Driver monitor started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    status_text = "No face detected"
    eyes_closed = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, w, h)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                eyes_closed = True
                status_text = "Eyes CLOSED"
            else:
                eyes_closed = False
                status_text = "Eyes OPEN"

    # Handle alarm timing
    if eyes_closed:
        if closed_eyes_start is None:
            closed_eyes_start = time.time()
        elif time.time() - closed_eyes_start > ALERT_TIME:
            status_text = "ALERT! Eyes closed too long!"
            play_alarm()
            closed_eyes_start = time.time()  # reset timer
    else:
        closed_eyes_start = None

    # Put status text on frame
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255) if "ALERT" in status_text else (0, 255, 0), 2)

    cv2.imshow("Driver Monitor", frame)

    # Quit if q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if alarm_playing is not None and alarm_playing.is_playing():
            alarm_playing.stop()
        break

cap.release()
cv2.destroyAllWindows()
