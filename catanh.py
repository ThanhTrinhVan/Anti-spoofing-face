import cv2
import time

def Mouse_event(event, x, y, f, img):
	if event == cv2.EVENT_LBUTTONDOWN:
		# su kien nhan chuot
		Mouse_event.x0 = x
		Mouse_event.y0 = y
		Mouse_event.isClick = True
	if event == cv2.EVENT_LBUTTONUP:
		# su kien tha chuot
		Mouse_event.x1 = x
		Mouse_event.y1 = y
		Mouse_event.isClick = False
		min_y = min(Mouse_event.y0, Mouse_event.y1)
		max_y = max(Mouse_event.y0, Mouse_event.y1)
		min_x = min(Mouse_event.x0, Mouse_event.x1)
		max_x = max(Mouse_event.x0, Mouse_event.x1)
		Mouse_event.img = img[min_y:max_y, min_x:max_x]
	if event == cv2.EVENT_MOUSEMOVE:
		Mouse_event.x = x
		Mouse_event.y = y

Mouse_event.img = None # bien cuc bo cua ham Mouse_event, co the goi gia tri o ngoai ham
Mouse_event.x0 = 0
Mouse_event.y0 = 0
Mouse_event.x1 = 0
Mouse_event.y1 = 0
Mouse_event.x = 0
Mouse_event.y = 0
Mouse_event.isClick = False

video = cv2.VideoCapture(0)
fps   = video.get(cv2.CAP_PROP_FPS) # so khung hinh tren giay (s)
wait_time = 1000/fps # thoi gian doi giua cac khung hinh (ms)
id = 0
while True:
	prev_time = time.time()

	ret, img = video.read()
	if not ret:
		break
	img_clone = img.copy()

	if Mouse_event.isClick:
		cv2.rectangle(img_clone, (Mouse_event.x0, Mouse_event.y0), (Mouse_event.x, Mouse_event.y), (0,0,255), 2)

	if Mouse_event.img is not None:
		cv2.imwrite("F:/7th/Medvedev/AntiSpoofingFaceID/face_identification/dataset/" + str(id) + ".jpg", Mouse_event.img)
		id = id + 1

	cv2.imshow("Video", img_clone)
	cv2.setMouseCallback("Video", Mouse_event, img_clone)

	delta_time = (time.time() - prev_time)*1000
	if delta_time > wait_time:
		delay_time = 1
	else:
		delay_time = wait_time - delta_time

	if cv2.waitKey(int(delay_time)) == ord('q'):
		break

cv2.destroyAllWindows()
