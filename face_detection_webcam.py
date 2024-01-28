import cv2

casc_path_frontal_face = "xml/haarcascade_frontalface_default.xml"
frontal_face_casc = cv2.CascadeClassifier(casc_path_frontal_face)

video_capture = cv2.VideoCapture(0)

def face_detection(casc_model, img_gray_scale):
    # Detection frontal faces
    faces_detected = casc_model.detectMultiScale(
        img_gray_scale,
        scaleFactor=1.5,
        minNeighbors=7,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    qtd_faces = len(faces_detected)

    if qtd_faces > 0:
        print("Detected {0} faces!".format(qtd_faces))

    return faces_detected

def draw_rectangle(faces):

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

while True:
    ret, frame = video_capture.read()

    img_gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frontal_faces = face_detection(frontal_face_casc, img_gray_scale)
    draw_rectangle(frontal_faces)

    cv2.imshow('face detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
