import cv2
import os
import glob
import numpy as np
import pygame
import threading
import time
import datetime as dt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, select
from sqlalchemy import text as sql_text
from sqlalchemy.orm import sessionmaker
from face_recognition.data_capsule import FaceData, EventData
from ultralytics import YOLO

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128

sound_playing_flag = False

def play_sounds(name, now, group):
    global sound_playing_flag
    sound_playing_flag = True

    if group == "staff2":
        greeting_voice = "sounds/hello_e.mp3"
    else:
        if now < dt.time(11,0,0):
            greeting_voice = "sounds/goodmorning.mp3"

        elif now > dt.time(18,0,0):
            greeting_voice = "sounds/goodevening.mp3"
        else:
            greeting_voice = "sounds/hello.mp3"
    name_calling_voice = f"sounds/{name}.mp3"
    mp3_files = [greeting_voice, name_calling_voice]
    for mp3 in mp3_files:
        pygame.mixer.music.load(mp3)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
    print("Sound play finished.")
    sound_playing_flag = False

# 特徴を辞書と比較してマッチしたユーザーとスコアを返す関数
#def match(recognizer, feature1, dictionary):
#    for element in dictionary:
#        user_id, feature2 = element
#        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
#        if score > COSINE_THRESHOLD:
#            return True, (user_id, score)
#    return False, ("", 0.0)
def match(session, feature1):
    items = session.scalars(select(FaceData).order_by(FaceData.feature_vector.cosine_distance(feature1.reshape(-1))).limit(1))
    for item in items:
        score = cosine_similarity(item.feature_vector.reshape(1,-1), feature1)
        return True, (item, score[0,0])

def main():
    # 特徴を読み込む
#    df_id = pd.read_csv("data/id2map.csv")
    engine_face = create_engine(os.environ.get("DATABASE_URL", "postgresql://localhost:5432/faces"))
    SessionClass_face = sessionmaker(engine_face)
    session_face = SessionClass_face()
    session_face.execute(sql_text('CREATE EXTENSION IF NOT EXISTS vector'))

    engine_event = create_engine(os.environ.get("DATABASE_URL", "postgresql://localhost:5432/faces"))
    SessionClass_event = sessionmaker(engine_event)
    session_event = SessionClass_event()

    yolo_model = YOLO('yolov8n.pt')
    yolo_target_name = "person"
    yolo_confidense_threshold = 0.0

    global sound_playing_flag
    sound_playing_flag = False
    pygame.mixer.init()
    print("Mixer inited.")    
    dictionary = []
#    files = glob.glob(os.path.join("./face_data/", "*.npy"))
#    for file in files:
#        feature = np.load(file)
#        user_id = os.path.splitext(os.path.basename(file))[0]
#        dictionary.append((user_id, feature))

    # モデルを読み込む
    weights = "onnx/yunet_n_640_640.onnx"
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    weights = "onnx/face_recognizer_fast.onnx"
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    cap = cv2.VideoCapture(0)

#    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    face_count = {}
    face_total_score = {}

    person_count = 0

    sound_played_flag = {}

    id_success, image = cap.read()
    while id_success:
#        image = cv2.resize(image, dsize=(640,360))
#        image = image[240:720,320:960]
        image_out = image.copy()
        now = dt.datetime.now()
        now_time = now.time()
    #    image = cv2.resize(image, None, fx=0.25, fy=0.25)
        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

#        yolo_result = yolo_model(cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB))
#
#        if yolo_result is not None:
#            boxes = yolo_result[0].boxes
#            for box in boxes:
#                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
#                confidense = box.conf[0]
#                cls = int(box.cls[0])
#                name = yolo_model.names[cls]
#
#                if name == yolo_target_name and confidense > yolo_confidense_threshold:
#                    cv2.rectangle(image, (xmin, ymin, xmax, ymax), (255, 0, 0), 3)
#                    person_count += 1



        # 入力サイズを指定する
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        # 顔を検出する
        result, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        if len(faces) == 0:
            face_count = {}
            face_total_score = {}

        for face in faces:
            # 顔を切り抜き特徴を抽出する
            aligned_face = face_recognizer.alignCrop(image, face)

            box = list(map(int, face[:4]))
            if  box[3] > 60 and box[2] > 40:
#            if  box[3] > 40 and box[2] > 26:
#            if True:
                feature = face_recognizer.feature(aligned_face)

                # 辞書とマッチングする
                result, user = match(session_face, feature)

                userdata, score = user
                score = float(score)

                # 顔のバウンディングボックスを描画する
                color = (0, 255, 0) if score > COSINE_THRESHOLD else (0, 0, 255)
                thickness = 2
                cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

#                if score > COSINE_THRESHOLD:
                if True:
                    id = userdata.id
                    name = userdata.name
#                    name = df_id["name"].iloc[int(id)]
#                    group = df_id["type"].iloc[int(id)]
#                    data = session_face.query(FaceData).where(FaceData.id == id)
                    #print(data)
                    group_id = userdata.group_id
                    id_name = f"{id}_{name}"
                    if group_id == 1:
                        group = "Staff"
                    elif group_id == 2:
                        group = "Staff2"
                    else:
                        group = None
                else:
                    name = "unknown"
                    id = None
                    group = None
                    id_name = name
                
                if score < COSINE_THRESHOLD:
                    display_name = "unknown"
                else:
                    display_name = id_name

                text = "{0} ({1:.2f})".format(display_name, score)
                position = (box[0], box[1] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 2.0
                cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

                if not sound_playing_flag:
                    if not face_count.get(id_name):
                        face_count[id_name] = 1
                        face_total_score[id_name] = score
                    else:
                        face_count[id_name] += 1
                        face_total_score[id_name] += score

                if face_count.get(id_name) and face_count[id_name] > 15:
                    average_score = face_total_score[id_name]/face_count[id_name]
                    item = EventData(detect_time=dt.datetime.now(), face_id=id, score=average_score)
                    session_event.add(item)
                    session_event.commit()

                    image_out_crop = image_out[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
                    cv2.imwrite(f"./recorded_faces/{item.id}.jpg", image_out_crop)

                    if average_score < COSINE_THRESHOLD:
                        print("Unknown face detected.")
#                         os.mak.edirs("./results/unknown/", exist_ok=True)
#                        cv2.imwrite("./results/unknown/{}.jpg".format(dt.datetime.strftime(now, "%Y%m%d%H%M%S")), image_out_crop)
    #                    cv2.imwrite("./results/{}.jpg".format(dt.datetime.strftime(now, "%Y%m%d%H%M%S")), image_out)
                    else:
                        print(f"Hello, {name}-san.")

#                        if not sound_played_flag.get(id_name) and not sound_playing_flag:
                        if not sound_playing_flag:
                            print("Sound play starting")
                            t = threading.Thread(target=play_sounds, args=(name, now_time,group,))
                            t.start()
                            sound_played_flag[id_name] = True
                        
                        average_score = face_total_score[id_name]/face_count[id_name]
#                        if average_score < 0.45:
#                        if average_score < 1.0:
#                            image_out_crop = image_out[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
#                            os.makedirs(f"./results/{id_name}/", exist_ok=True)
#                            cv2.imwrite("./results/{0}/{1}_{2:.2f}.jpg".format(id_name, dt.datetime.strftime(now, "%Y%m%d%H%M%S"), average_score), image_out_crop)
#                            with open("./results/{0}/{1}_{2:.2f}.txt".format(id_name, dt.datetime.strftime(now, "%Y%m%d%H%M%S"), average_score), "w") as fout:
#                                print(face_count, file=fout)
#                                print(face_total_score, file=fout)
    #                    cv2.rectangle(image_out, (0, 0), (400, 40), color=(255,255,255), thickness=-1)
    #                    cv2.putText(image_out, f"{id}: {face_total_score[id]/face_count[id]:.2f}", org=(0,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_4)
    #                    cv2.imwrite("./results/{}.jpg".format(dt.datetime.strftime(now, "%Y%m%d%H%M%S")), image_out)
                    face_count = {}
                    face_total_score = {}                

        if person_count > 15:
            cv2.imwrite("person_picture/{}.jpg".format(now.strftime("%Y%m%d%H%M%S")), image_out)
            person_count = 0

        if dt.time(6,0,0,0) < now_time < dt.time(6,0,1,0) and len(sound_played_flag) > 0:
            print("Dictionary re-initialized.")
            sound_played_flag = {}

        # 画像を表示する
        cv2.imshow("face recognition", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        id_success, image = cap.read()
    pygame.mixer.quit()
    print(sound_played_flag)
    
if __name__ == '__main__':
    main()