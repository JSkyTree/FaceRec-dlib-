import dlib, cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

#learn data
descs = np.load('descs/descs.npy')[()]

#stream url
url = "http://192.168.0.101:8091/?action=stream"

def encode_face(img):
  dets = detector(img, 1)

  if len(dets) == 0:
    return np.empty(0)

  for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    return np.array(face_descriptor)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
  exit()

_, img_bgr = cap.read() # (800, 1920, 3)
padding_size = 0
resized_width = 400
video_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1]))
output_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1] + padding_size * 2))



while True:
  ret, img_bgr = cap.read()
  if not ret:
    break

  img_bgr = cv2.resize(img_bgr, video_size)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  
  dets = detector(img_bgr, 1)



  for k, d in enumerate(dets):
    shape = sp(img_rgb, d)
    face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)

    last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

    

    for name, saved_desc in descs.items():
      dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

      if dist < last_found['dist']:
        last_found = {'name': name, 'dist': dist, 'color': (255,255,255)}

    
    
    
    cv2.rectangle(img_bgr, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
    cv2.putText(img_bgr, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)

  
  
  

  cv2.imshow('img', img_bgr)
  if cv2.waitKey(1) == ord('q'):
    break

cv2.destroyAllWindows()
cap.release()