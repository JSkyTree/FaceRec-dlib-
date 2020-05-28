import cv2

url = "http://192.168.0.101:8091/?action=stream"
cap = cv2.VideoCapture(url)

while True:
    #image read
    ret, image = cap.read()

    #image show
    cv2.imshow('stream', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
    
