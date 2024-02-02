from flask import Flask, render_template, Response, session
from utils import seguimientoManos as sm
import cv2
import mediapipe as mp
from flask_wtf import FlaskForm
import numpy as np
from tensorflow import keras
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
from utils import inferencia
from ultralytics import YOLO
from YOLO_Inferencia import video_deteccion

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lenguasen'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Creamos nuestra funcion de dibujo
mp_holistic = mp.solutions.holistic # Holistic model
mp_dibujo = mp.solutions.drawing_utils
ConfDibu = mp_dibujo.DrawingSpec(thickness=1, circle_radius=1)

# Creamos un objeto donde almacenaremos los puntos de rostro y manos
mp_rostro = mp.solutions.face_mesh
mp_mano = mp.solutions.hands
mp_pose = mp.solutions.pose
MallaFacial = mp_rostro.FaceMesh(max_num_faces=1)
Torso = mp_pose.PoseLandmark
Mano = mp_mano.HandLandmark

# Realizamos la Videocaptura
cap = cv2.VideoCapture(1)

#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")

def gen_frame():
    # Empezamos
    while True:
        # Leemos la VideoCaptura
        ret, frame = cap.read()

        # Si tenemos un error
        if not ret:
            break

        else:
            # Correccion de color
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Observamos los resultados
            resultados = MallaFacial.process(frameRGB)

            # Si tenemos rostros
            if resultados.multi_face_landmarks:
                # Iteramos
                for rostros in resultados.multi_face_landmarks:
                    # Dibujamos
                    mp_dibujo.draw_landmarks(frame, rostros, mp_rostro.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

            # Detectamos manos
            resultados_manos = Mano.process(frameRGB)

            # Si tenemos manos
            if resultados_manos.multi_hand_landmarks:
                # Iteramos
                for manos in resultados_manos.multi_hand_landmarks:
                    # Dibujamos
                    mp_dibujo.draw_landmarks(frame, manos, mp_mano.HAND_CONNECTIONS, ConfDibu, ConfDibu)

            # Detectamos torso
            resultados_torso = Torso.process(frameRGB)

            # Si tenemos torso
            if resultados_torso.torso_landmarks:
                # Dibujamos
                mp_dibujo.draw_landmarks(frame, resultados_torso.torso_landmarks, mp_pose.TORSO_CONNECTIONS, ConfDibu, ConfDibu)

            # Codificamos nuestro video en Bytes
            suc, encode = cv2.imencode('.jpg', frame)
            frame = encode.tobytes()

            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_deteccion(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/', methods=['GET','POST'])

@app.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('index.html')

@app.route('/alfabeto',methods=['GET','POST'])
def alfabeto():
    return render_template('alfabeto.html')

@app.route('/gestos',methods=['GET','POST'])
def gestos():
    return render_template('gestos.html')

@app.route('/diccionario')
def diccionario():
    return render_template('diccionario.html')

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    # Draw face connections
    mp_dibujo.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_dibujo.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_dibujo.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Dibujar pose conecciones
    mp_dibujo.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_dibujo.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_dibujo.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_dibujo.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_dibujo.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_dibujo.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_dibujo.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_dibujo.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_dibujo.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame

modelg = keras.models.load_model('models/gestos.h5')
actions = np.array(['hola', 'adios','gracias','perdon','teAmo', 'porfavor'])

@app.route('/detecciongesto', methods=['GET','POST'])
def generate_frame():
    def generate():
        # Initialize variables
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()

                # Make detections and draw landmarks
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                # Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = modelg.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))

                    # Viz logic
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            sentence.append(actions[np.argmax(res)])
                            sentence = sentence[-5:]

                    image = prob_viz(res, actions, image, colors)

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

#cargar modelo
modelY = YOLO("models/lenguasen.pt")

#Crear objeto detector de manos
detector = sm.detectormanos(Confdeteccion=0.8)

# Función para la detección de gestos en tiempo real
def deteccion_abc():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        frame = detector.encontrarmanos(frame, dibujar=False)
        lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False, color=[0, 255, 0])

        if mano == 1:
            xmin, ymin, xmax, ymax = bbox
            xmin = xmin - 40
            ymin = ymin - 40
            xmax = xmax + 40
            ymax = ymax + 40

            recorte = frame[ymin:ymax, xmin:xmax]
            recorte = cv2.resize(recorte, (720, 720), interpolation=cv2.INTER_CUBIC)

            # Realiza acciones adicionales con el recorte, si es necesario

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_web(path_x= session.get('video_path')), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/deteccionabc')
def deteccionabc():
    return render_template('deteccionabc.html')

if __name__ == '__main__':
    app.run(debug=True)