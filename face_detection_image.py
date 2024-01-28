import cv2
import os


def face_detection(casc_model, image_gray_scale):
    # Detection frontal faces
    faces_detected = casc_model.detectMultiScale(
        image_gray_scale,
        scaleFactor=1.5,
        minNeighbors=7,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    qtd_faces = len(faces_detected)

    if qtd_faces > 0:
        print("Detected {0} faces!".format(qtd_faces))

    return faces_detected, qtd_faces


def draw_rectangle(faces, image):

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


casc_path_frontal_face = "xml/haarcascade_frontalface_default.xml"
frontal_face_casc = cv2.CascadeClassifier(casc_path_frontal_face)

files_in_path = os.listdir("images/")
images = [image for image in files_in_path
                if image.endswith(".jpeg")]

for image in images:
    image = cv2.imread(f"images/{ image }")

    image_gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    frontal_faces, qtd_faces = face_detection(frontal_face_casc, image_gray_scale)
    draw_rectangle(frontal_faces, image)

    cv2.imshow(f'{ qtd_faces } face(s) detected', image)
    cv2.waitKey(0)
