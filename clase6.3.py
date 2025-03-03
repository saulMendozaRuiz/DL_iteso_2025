import cv2

cap = cv2.VideoCapture(0)

# Automatically grab width and height from video feed
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter('/home/saulmendoza/Pictures/my_capture.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (width, height))

while True:
# Capture frame-by-frame
    ret, frame = cap.read()
# Write the video
    writer.write(frame)
# Display the resulting frame
    cv2.imshow('frame',frame)
# This command let's us quit with the "q" button on a keyboard.
# Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
writer.release()

cv2.destroyAllWindows()
