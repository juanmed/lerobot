import cv2
import threading
import numpy as np

DEVICES = ["/dev/video0", "/dev/video2", "/dev/video4", "/dev/video6"]
frames = [None] * len(DEVICES)
lock = threading.Lock()
running = True


def capture_loop(idx, device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Failed to open {device}")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Flush stale buffers after format negotiation
    for _ in range(5):
        cap.grab()
    while running:
        ret, frame = cap.read()
        if ret:
            with lock:
                frames[idx] = frame
        else:
            print(f"Failed to read frame from {device}")
    cap.release()


threads = []
for i, dev in enumerate(DEVICES):
    t = threading.Thread(target=capture_loop, args=(i, dev), daemon=True)
    t.start()
    threads.append(t)

print("Press 'q' to quit.")
while True:
    with lock:
        current = frames[:]

    placeholders = []
    for i, frame in enumerate(current):
        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"No signal: {DEVICES[i]}", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            placeholders.append(placeholder)
        else:
            placeholders.append(frame)

    combined = np.hstack(placeholders)
    cv2.imshow("Camera Streams", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
cv2.destroyAllWindows()
