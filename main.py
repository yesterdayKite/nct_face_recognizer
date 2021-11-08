

import face_recognition
import cv2
import numpy as np


"""
# face_recognition 오픈소스를 이용한 
  아이돌그룹 NCT 얼굴 구분 프로그램
"""


# 태일
TaeIl_image = face_recognition.load_image_file("personal_face/TaeIl.jpeg")
TaeIl_face_encoding = face_recognition.face_encodings(TaeIl_image)[0]

# 쟈니
Johnny_image = face_recognition.load_image_file("personal_face/Johnny.jpeg")
Johnny_face_encoding = face_recognition.face_encodings(Johnny_image)[0]

# 태
TaeYong_image = face_recognition.load_image_file("personal_face/TaeYong.jpeg")
TaeYong_face_encoding = face_recognition.face_encodings(TaeYong_image)[0]

# 유타
Yuta_image = face_recognition.load_image_file("personal_face/Yuta.jpeg")
Yuta_face_encoding = face_recognition.face_encodings(Yuta_image)[0]

# 쿤
Kun_image = face_recognition.load_image_file("personal_face/Kun.jpeg")
Kun_face_encoding = face_recognition.face_encodings(Kun_image)[0]

# 도
DoYoung_image = face_recognition.load_image_file("personal_face/DoYoung.jpeg")
DoYoung_face_encoding = face_recognition.face_encodings(DoYoung_image)[0]

#텐
Ten_image = face_recognition.load_image_file("personal_face/Ten.jpeg")
Ten_face_encoding = face_recognition.face_encodings(Ten_image)[0]

#재현
JaeHyun_image = face_recognition.load_image_file("personal_face/JaeHyun.jpeg")
JaeHyun_face_encoding = face_recognition.face_encodings(JaeHyun_image)[0]

# 윈윈
WinWin_image = face_recognition.load_image_file("personal_face/WinWin.jpeg")
WinWin_face_encoding = face_recognition.face_encodings(WinWin_image)[0]

# 정우
JungWoo_image = face_recognition.load_image_file("personal_face/JungWoo.jpeg")
JungWoo_face_encoding = face_recognition.face_encodings(JungWoo_image)[0]

# 루카스
Lucas_image = face_recognition.load_image_file("personal_face/Lucas.jpeg")
Lucas_face_encoding = face_recognition.face_encodings(Lucas_image)[0]

# 마크
Mark_image = face_recognition.load_image_file("personal_face/Mark.jpeg")
Mark_face_encoding = face_recognition.face_encodings(Mark_image)[0]

# 샤우쥔
XiaoJun_image = face_recognition.load_image_file("personal_face/XiaoJun.jpeg")
XiaoJun_face_encoding = face_recognition.face_encodings(XiaoJun_image)[0]

# 헨드리
Hendrey_image = face_recognition.load_image_file("personal_face/Hendrey.jpeg")
Hendrey_face_encoding = face_recognition.face_encodings(Hendrey_image)[0]

# 런쥔
RenJun_image = face_recognition.load_image_file("personal_face/RenJun.jpeg")
RenJun_face_encoding = face_recognition.face_encodings(RenJun_image)[0]

# 제노
Jeno_image = face_recognition.load_image_file("personal_face/Jeno.jpeg")
Jeno_face_encoding = face_recognition.face_encodings(Jeno_image)[0]

# 해찬
HaeChan_image = face_recognition.load_image_file("personal_face/HaeChan.jpeg")
HaeChan_face_encoding = face_recognition.face_encodings(HaeChan_image)[0]

# 재민
JaeMin_image = face_recognition.load_image_file("personal_face/JaeMin.jpeg")
JaeMin_face_encoding = face_recognition.face_encodings(JaeMin_image)[0]

# 양양
YangYang_image = face_recognition.load_image_file("personal_face/YangYang.jpeg")
YangYang_face_encoding = face_recognition.face_encodings(YangYang_image)[0]

# 쇼타로
ShoTaRo_image = face_recognition.load_image_file("personal_face/ShoTaRo.jpeg")
ShoTaRo_face_encoding = face_recognition.face_encodings(ShoTaRo_image)[0]

# 성찬
SungChan_image = face_recognition.load_image_file("personal_face/SungChan.jpeg")
SungChan_face_encoding = face_recognition.face_encodings(SungChan_image)[0]

# 천러
#ChenLe_image = face_recognition.load_image_file("personal_face/ChenLe.jpeg")
#ChenLe_face_encoding = face_recognition.face_encodings(ChenLe_image)[0]

# 지성
JiSung_image = face_recognition.load_image_file("personal_face/JiSung.jpeg")
JiSung_face_encoding = face_recognition.face_encodings(JiSung_image)[0]


known_face_encodings = [
    TaeIl_face_encoding,
    Johnny_face_encoding,
    TaeYong_face_encoding,
    Yuta_face_encoding,
    Kun_face_encoding,
    DoYoung_face_encoding,
    Ten_face_encoding,
    JaeHyun_face_encoding,
    WinWin_face_encoding,
    JungWoo_face_encoding,
    Lucas_face_encoding,
    Mark_face_encoding,
    XiaoJun_face_encoding,
    Hendrey_face_encoding,
    RenJun_face_encoding,
    Jeno_face_encoding,
    HaeChan_face_encoding,
    JaeMin_face_encoding,
    YangYang_face_encoding,
    ShoTaRo_face_encoding,
    SungChan_face_encoding,
    #ChenLe_face_encoding,
    JiSung_face_encoding
]
known_face_names = [
    "TaeIl",
    "Johnny",
    "TaeYong",
    "Yuta",
    "Kun",
    "DoYoung",
    "Ten",
    "JaeHyun",
    "WinWin",
    "JungWoo",
    "Lucas",
    "Mark",
    "XiaoJun",
    "Hendrey",
    "RenJun",
    "Jeno",
    "HaeChan",
    "JaeMin",
    "YangYang",
    "ShoTaRo",
    "SungChan",
    #"ChenLe",
    "JiSung"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

thumbnail = cv2.imread('thumbnail.png', cv2.IMREAD_COLOR)
cv2.imshow('thumbnail', thumbnail)

video_capture = cv2.VideoCapture(0)


while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    #얼굴이 감지되었을때
    if process_this_frame:
        # 모든 얼굴과 위치를 찾는다
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 해당하는 얼굴이 없으면 unknown 처
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # 가장 적게 차이 나는 얼굴을 매칭시킨
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # 결과 송출
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 박스 그리기
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 라벨 그리기
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()










