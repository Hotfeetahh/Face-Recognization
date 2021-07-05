import keras
import cv2
import os
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

fCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
images = ['4.91.jpg', '2.68.jpg', '2.165.jpg', '2.99.jpg', '4.74.jpg', '2.140.jpg']

def ref(dd):
    if (dd[0][0] >= dd[0][1]):
        return (0, dd[0][0])
    else:
        return (1, dd[0][1])

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['xoan', 'nhu']
model = keras.models.load_model('my_model')
for i in images:
    im = cv2.imread(i)
    imgGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = fCascade.detectMultiScale(imgGray, scaleFactor=1.05,
                                      minNeighbors=10,
                                      minSize=(150, 150),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cropImg = im[y:y+h, x:x+w]
        img = cv2.resize(cropImg, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0)
        dd = model.predict(img)
        print(dd)
        id, confidence = ref(dd)
        print(id, confidence)
        if (confidence < 1 and confidence > 0.55):
            id = names[id]
            confidence = "  {0}%".format(round(100*(confidence)))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100*(confidence)))

        cv2.putText(
            im,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            im,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('img', im)
    cv2.waitKey(0)


# cam ------------------------------
# recognizer = keras.models.load_model('my_model')
# cascadePath = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascadePath);
# font = cv2.FONT_HERSHEY_SIMPLEX
# # iniciate id counter
# id = 0
# # names related to ids: example ==> Marcelo: id=1,  etc
# names = ['xoan', 'nhu']
# # Initialize and start realtime video capture
# cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # set video widht
# cam.set(4, 480)  # set video height
# # Define min window size to be recognized as a face
# minW = 0.1 * cam.get(3)
# minH = 0.1 * cam.get(4)
# while True:
#     ret, img = cam.read()
#     img = cv2.flip(img, -1)  # Flip vertically
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=5,
#         minSize=(int(minW), int(minH)),
#     )
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#
#         # If confidence is less them 100 ==> "0" : perfect match
#         if (confidence < 100):
#             id = names[id]
#             confidence = "  {0}%".format(round(100 - confidence))
#         else:
#             id = "unknown"
#             confidence = "  {0}%".format(round(100 - confidence))
#
#         cv2.putText(
#             img,
#             str(id),
#             (x + 5, y - 5),
#             font,
#             1,
#             (255, 255, 255),
#             2
#         )
#         cv2.putText(
#             img,
#             str(confidence),
#             (x + 5, y + h - 5),
#             font,
#             1,
#             (255, 255, 0),
#             1
#         )
#
#     # cv2.imshow('camera', img)
#     k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
#     if k == 27:
#         break
# # Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
# cam.release()
# cv2.destroyAllWindows()

# Img read--------------------------------
# recognizer = keras.models.load_model('my_model')
# cascadePath = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascadePath)
# font = cv2.FONT_HERSHEY_SIMPLEX
# # iniciate id counter
# id = 0
# # names related to ids: example ==> Marcelo: id=1,  etc
# names = ['xoan', 'nhu']
#
# images = ['4.91.jpg', '2.68.jpg', '2.165.jpg', '2.99.jpg', '4.74.jpg', '2.140.jpg']
# minW, minH = 150, 150
# for i in images:
#     # ret, img = cam.read()
#     img = cv2.imread(i)
#     img = cv2.flip(img, -1)  # Flip vertically
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=5,
#         minSize=(int(minW), int(minH)),
#     )
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         tImg = img[y:y + h, x:x + w]
#         tImg = cv2.resize(tImg, (224, 224), interpolation=cv2.INTER_AREA)
#         tImg = np.expand_dims(tImg, axis=0)
#         id, confidence = recognizer.predict(tImg)
#
#         # If confidence is less them 100 ==> "0" : perfect match
#         if (confidence < 100):
#             id = names[id]
#             confidence = "  {0}%".format(round(100 - confidence))
#         else:
#             id = "unknown"
#             confidence = "  {0}%".format(round(100 - confidence))
#
#         cv2.putText(
#             img,
#             str(id),
#             (x + 5, y - 5),
#             font,
#             1,
#             (255, 255, 255),
#             2
#         )
#         cv2.putText(
#             img,
#             str(confidence),
#             (x + 5, y + h - 5),
#             font,
#             1,
#             (255, 255, 0),
#             1
#         )
#
#     cv2.imshow('camera', img)
#     k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
#     if k == 27:
#         break
# # Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
# cv2.destroyAllWindows()
