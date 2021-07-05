import os
import cv2
# from PIL import Image

def process(img):
    global fCascade
    global oP
    global cnt
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = fCascade.detectMultiScale(imgGray, 1.1, 15)
    if len(faces) > 1:
        mxSize = 0
        id = 0
        for i in range(0, len(faces)):
            w = faces[i][2]
            h = faces[i][3]
            if (w * h > mxSize):
                mxSize = w * h
                id = i

        x, y, w, h = list(map(int, faces[id]))
        if (w >= 128 or h >= 128):
            cropImg = imgGray[y:y + h, x:x + w]
            oImg = cv2.resize(cropImg, (224, 224), interpolation=cv2.INTER_AREA)
            # print(os.path.join(oP, (str(cnt) + '.jpg')))
            cv2.imwrite(os.path.join(oP, (str(cnt) + '.jpg')), oImg)
            # print(cnt)
            cnt += 1

    elif(len(faces) == 1):
        x, y, w, h = list(map(int, faces[0]))
        if (w >= 128 or h >= 128):
            cropImg = imgGray[y:y + h, x:x + w]
            oImg = cv2.resize(cropImg, (224, 224), interpolation=cv2.INTER_AREA)
            # print(os.path.join(oP, (str(cnt) + '.jpg'))
            cv2.imwrite(os.path.join(oP, (str(cnt) + '.jpg')), oImg)
            # print(cnt)
            cnt += 1



inpPath = "./dataset"
outPath = "./dts"
fCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

inpClasses = os.listdir(inpPath)
# chinh sua anh tu dataset cu roi luu vao directory cua dataset moi
for inpClass in inpClasses:
    p = os.path.join(inpPath, inpClass)
    oP = os.path.join(outPath, inpClass)
    k = os.listdir(p)

    print(p, len(k))
    cnt = 0
    for i in k:
        imgPath = os.path.join(p, i)
        img = cv2.imread(imgPath)

        process(img)
    print(cnt)

# kiem tra shape cua anh trong dataset moi
# outClasses = os.listdir(outPath)
# for outClass in outClasses:
#     p = os.path.join(outPath, outClass)
#     k = os.listdir(p)
    # print(k)
    # for i in k:
    #     img = cv2.imread(os.path.join(p, i))
        # print(img.shape)
