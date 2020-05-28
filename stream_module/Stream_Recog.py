import dlib, cv2
import numpy as np
from stream_module import imagezmq


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

#learn data
descs = np.load('descs/descs.npy')[()]

def save_name(name, dist):
    name_find = open("texts/"+str(name)+".txt", 'w')
    dist_find = round(float(dist),4)
    name_find.write(str(dist_find))
    name_find.close()

def image_resize(cap, padding, width):
    _, img_bgr = cap.read()
    padding_size = padding
    resized_width = width
    video_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1]))
    output_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1] + padding_size * 2))

    return video_size, output_size


def start_stream():

    image_hub = imagezmq.ImageHub()

    while True:
        rpi_name, img_bgr = image_hub.recv_image()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        dets = detector(img_bgr, 1)

        #find face and rec
        for k, d in enumerate(dets):
          shape = sp(img_rgb, d)
          face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)

          last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

          for name, saved_desc in descs.items():
            dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

            if dist < last_found['dist']:
              last_found = {'name': name, 'dist': dist, 'color': (255,255,255)}

              save_name(name, dist)        
        
          #drow box
          cv2.rectangle(img_bgr, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
          cv2.putText(img_bgr, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)
        
        cv2.imshow(rpi_name, img_bgr)
        if cv2.waitKey(1) == ord('q'):
          break

        image_hub.send_reply(b'OK')


    cv2.destroyAllWindows()

