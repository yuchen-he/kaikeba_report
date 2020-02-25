import cv2

video_orig = cv2.VideoCapture('./arduino2.mp4')
count = 1

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('arduino_resized.mp4',fourcc, 30.0, (1280, 720))

while(True):
    ret, frame = video_orig.read()
    new_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
    if count == 1:
    #if count == 50:
        cv2.imwrite('./first_frame_arduino.jpg', new_frame)

    if count >= 1:
        out.write(new_frame)
    count += 1
    #cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
video_orig.release()
out.release()
cv2.destroyAllWindows()
