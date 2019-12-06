import ResNet
import argparse
import cv2
import time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-d", "--draw", type=bool, default=False,
                help="draw dots on face")
ap.add_argument("-n", "--attempt", type=int, default=10,
                help="number of attempts")
args = vars(ap.parse_args())

protoPath = "./face_detector/deploy.prototxt"
modelPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
netFaceDet = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

protoPath2 = "./face_alignment/2_deploy.prototxt"
modelPath2 = "./face_alignment/2_solver_iter_800000.caffemodel"
netFaceDet2 = cv2.dnn.readNetFromCaffe(protoPath2, modelPath2)

def crop_by_mark(image, landmark):
    scale = 3.5
    image_size = 224
    ct_x, std_x = landmark[:, 0].mean(), landmark[:, 0].std()
    ct_y, std_y = landmark[:, 1].mean(), landmark[:, 1].std()

    std_x, std_y = scale * std_x, scale * std_y

    src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
    dst = np.float32([((image_size - 1) / 2.0, (image_size - 1) / 2.0),
                      ((image_size - 1), (image_size - 1)),
                      ((image_size - 1), (image_size - 1) / 2.0)])
    retval = cv2.getAffineTransform(src, dst)
    result = cv2.warpAffine(image, retval, (image_size, image_size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    return result

def check_real_fake(img):
    data = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
    data = data[np.newaxis, :]
    data = torch.FloatTensor(data)
    with torch.no_grad():
        outputs = modelNetRF(data)
        outputs = torch.softmax(outputs, dim=-1)
        preds = outputs.to('cpu').numpy()
        fake_prob = preds[:, FAKE]
    return  fake_prob

if __name__ == '__main__':
    print("[INFO] Load ResNet anti spoofing...")
    net_path = "a8.pth"
    modelNetRF = getattr(ResNet, "MyresNet18")().eval()
    modelNetRF.load(net_path)
    modelNetRF.train(False)

    FAKE = 1
    thresh_probability = 0.9 #0.85
    REAL = 0

    print("[INFO] Load face recognizer...")
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read('./face_identification/trained.yml')

    print("[INFO] Start video stream....")
    vd = cv2.VideoCapture(0)
    numIncorrect = 0
    while True:
        ret, frame = vd.read()
        if ret is None:
            break
        # Face detection
        timeStart = time.time()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        netFaceDet.setInput(blob)
        detectionFaces = netFaceDet.forward()
        timeEnd = time.time()
        # print('Detect times : %.3f ms' % ((timeEnd - timeStart) * 1000))
        for i in range(0, detectionFaces.shape[2]):
            confidence = detectionFaces[0, 0, i, 2]
            if confidence > args["confidence"]:
                # Bounding detection face from frame
                box = detectionFaces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (sx, sy, ex, ey) = (startX, startY, endX, endY)
                # Find borders of face: Loai bo phan tran, lay phan mat, tai
                ww = (endX - startX) // 10
                hh = (endY - startY) // 5
                startX = startX - ww
                startY = startY + hh
                endX = endX + ww
                endY = endY + hh
                # If startX, Y and endX, y are not in range of frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                # Find dots on Face (dots are eye, noise, lip ...)
                x1 = int(startX)
                y1 = int(startY)
                x2 = int(endX)
                y2 = int(endY)
                # Crop Face
                roi = frame[y1:y2, x1:x2]
                grayFace = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                # Histogram of brightness
                matrixFace = np.float32(grayFace)
                m = np.zeros((40, 40))
                sd = np.zeros((40, 40))
                mean, std_dev = cv2.meanStdDev(matrixFace, m, sd) # gives you a mean for each channel and a standard deviation for each channel as arrays
                new_mean = mean[0][0]
                new_std = std_dev[0][0]
                matrixFace = (matrixFace - new_mean) / (0.000001 + new_std)
                # cv2.imshow("Brightness Face", matrixFace)
                blob = cv2.dnn.blobFromImage(cv2.resize(matrixFace, (40, 40)), 1.0, (40, 40), (0, 0, 0))
                netFaceDet2.setInput(blob)
                alignFace = netFaceDet2.forward()
                aligns = [] # list of dots on one face
                alignss = []
                # Calculate coordinate of dots on Face
                for k in range(0, 68):
                    dotTMP = []
                    x = alignFace[0][2 * k] * (x2 - x1) + x1
                    y = alignFace[0][2 * k + 1] * (y2 - y1) + y1
                    if (args["draw"] == True):
                        cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
                    dotTMP.append(int(x))
                    dotTMP.append(int(y))
                    aligns.append(dotTMP)
                cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
                alignss.append(aligns)

                mark = np.asarray(alignss, dtype=np.float32)
                mark = mark[np.argsort(np.std(mark[:,:,1], axis=1))[-1]]
                img = crop_by_mark(frame, mark)
                fake_probability = check_real_fake(img)
                if fake_probability > thresh_probability:
                    cv2.putText(frame, "Fake", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)
                else:
                    gray = cv2.cvtColor(frame[sy:ey, sx:ex], cv2.COLOR_RGB2GRAY)
                    id, accuracy = faceRecognizer.predict(gray)
                    if accuracy <= 50:
                        print("Correct! System unlocked!!!")
                        cv2.putText(frame, "Thanh - Real", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)
                        numIncorrect = 0
                    else:
                        print("Incorrect! [", numIncorrect, "]")
                        numIncorrect = numIncorrect + 1;
                        if numIncorrect > args["attempt"]:
                            print("System locked! Because number of attempts exceeds the limit.")
                            print("Please wait in 5 minutes")
                            #cv2.waitKey(10000)
                            numIncorrect = 0
                        cv2.putText(frame, "Unknown - Real", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("[INFO] End process...")
    cv2.destroyAllWindows()