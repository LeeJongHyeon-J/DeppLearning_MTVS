import os
import base64
import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np
import av
import torch
import tempfile
import time
import subprocess


torch.classes.__path__ = [os.path.join(torch.__path__[0], "classes")]
os.environ["STREAMLIT_WATCH_FILE"] = "false"

# YOLO 모델 로드
model = YOLO("best.pt")  # 사전 학습된 모델을 사용

# 페이지 설정
st.set_page_config(page_title="드론으로 생명을 살리는 감지 시스템",  layout="wide") #page_icon="data/Brone.png",

# 배경 설정 함수
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 배경 이미지 설정
# set_background("data/drone.png")

# 사이드바 메뉴
with st.sidebar:
    st.sidebar.image("./image/MeaMi.png")
    # st.image("data/logo.png", width=120)
    st.title("분석 모델 선택")
    page = st.selectbox("이동할 섹션을 선택하세요:", ["홈", "실시간 탐지","이미지 분석", "영상 분석", "영상 즉시 분석"], key="sidebar_select")
    st.sidebar.markdown(""" 
        8조
        - 손영석
        - 이종현
        - 배성우
        - 박범기

        ---
        - 사용모델: YOLOv11n
        - 데이터 수집처: roboflow
        - 웹 제작: streamlit
    """)

# 새로고침 시 홈으로
if 'page' not in st.session_state:
    st.session_state.page = "홈"

if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()

# 메인 페이지 제목
st.markdown(f"<h2 style='text-align:center;'>🚀 드론을 활용한 실종자 수색 시스템</h2>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size:18px; font-weight:bold;'>📌 현재 페이지: {page}</div>", unsafe_allow_html=True)


# 공통 YOLO 탐지 함수 (박스 정보 반환)
def detect_and_draw(image):
    results = model(image)
    result = results[0]
    boxes = result.boxes
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    xyxy = boxes.xyxy.tolist()

    detections = []
    for i, box in enumerate(xyxy):
        if int(class_ids[i]) == 0:  # 0번 클래스: 사람
            x1, y1, x2, y2 = map(int, box)
            conf = confidences[i]
            detections.append((x1, y1, x2, y2, conf))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image, detections

# 박스 그리기만 하는 함수
def draw_boxes(image, detections):
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'person {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


# 추가//실시간 탐지
frame_count = 0
def process_frame(frame):
    global frame_count
    img = frame.to_ndarray(format="bgr24")
    # if frame_count % 5 == 0:
    result = detect_and_draw(img)[0]
    process_frame.last_result = result
    frame_count += 1
    return av.VideoFrame.from_ndarray(process_frame.last_result, format="bgr24")
process_frame.last_result = None


# 동영상 코덱 문제 처리 함수
def convert_to_h264(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-acodec", "aac", output_path
    ]
    subprocess.run(command, check=True)


# 사진 갤러리 처리
if page == "이미지 분석":
    st.title("📷 이미지 분석")
    uploaded_image = st.file_uploader("📤 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        processed_img = detect_and_draw(open_cv_image)[0]
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)


# 동영상 갤러리 처리
elif page == "영상 분석":
    st.title("🎞️ 영상 분석")
    uploaded_video = st.file_uploader("📤 영상을 업로드하세요", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

        progress_bar = st.progress(0, text="탐지 중입니다. 잠시만 기다려 주세요.")

        frame_interval = round(fps*0.1)
        frame_idx = 0
        last_detections = []  # 마지막 탐지 결과 저장

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            if frame_idx % frame_interval == 0:
                processed, detections = detect_and_draw(frame)
                last_detections = detections  # 최신 탐지 결과 저장
            else:
                # 이전 탐지 결과로 박스만 그림
                processed = draw_boxes(frame.copy(), last_detections)
            out.write(processed)

            percent_complete = int((frame_idx + 1) / total_frames * 100)
            progress_bar.progress(percent_complete, text=f"동영상 처리 중... ({percent_complete}%)")
            frame_idx += 1

            # time.sleep(0.001)  # 필요시만 사용

        cap.release()
        out.release()
        progress_bar.empty()

        # 변환 파일도 임시 파일로 생성
        converted_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        with st.spinner("인코딩 중입니다. 잠시만 기다려 주세요..."):
            convert_to_h264(out_file.name, converted_file.name)

        st.success("탐지가 완료되었습니다! 아래에서 결과 영상을 확인하세요.")

        # 변환된 파일을 바이너리로 읽어서 넘김
        with open(converted_file.name, "rb") as video_file:
            st.video(video_file.read())            


# 동영상 실시간 분석 후 반환
elif page == "영상 즉시 분석":
    st.title("🎞️ 영상 즉시 분석(낮은 FPS)")
    uploaded_video = st.file_uploader("📤 동영상을 업로드하세요", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        # 동영상 캡처
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stframe = st.empty()  # 실시간 프레임 표시용 placeholder

        start_time = time.time()
        
        frame_interval = round(fps*0.4) # 출력 프레임 간격
        frame_idx = 0
        last_detections = []  # 마지막 탐지 결과 저장

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            if frame_idx % frame_interval == 0:
                processed, detections = detect_and_draw(frame)
                last_detections = detections

                stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # fps에 맞춘 시간 조절
            elapsed_time = time.time() - start_time
            expected_time = frame_idx / fps
            if elapsed_time < expected_time:
                time.sleep(expected_time - elapsed_time)

            frame_idx += 1

        cap.release()


elif page == "실시간 탐지":
    st.title("실시간 카메라 탐지")
    st.write("카메라와 연동하여 실시간으로 탐지합니다.")
    
    webrtc_streamer(
        key="realtime",
        video_frame_callback=process_frame,
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": "stun:stun.l.google.com:19302"},
                        {"urls": "stun:stun1.l.google.com:19302"},
                        {"urls": "stun:stun2.l.google.com:19302"},
                        {"urls": "stun:stun3.l.google.com:19302"},
                        {"urls": "stun:stun4.l.google.com:19302"},]
        }
    )

# 기타 페이지
elif page == "홈":
    st.markdown(
        """
        <h2 style='text-align: center;'>8조 딥러닝 프로젝트!</h2>
        """,
        
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style='text-align: center; font-size:18px;'>
        이 시스템은 드론이 촬영한 영상에서 사람으로 추정되는 물체를 포착합니다.
        </p>
        """,
        unsafe_allow_html=True
    )

    col1,col2 = st.columns(2)

    
    with col1:
    # 이미지 중앙 정렬
        

        with open("./image/딥러닝프로젝트.png", "rb") as img_file:
            img_bytes = img_file.read()
            encoded = base64.b64encode(img_bytes).decode()

        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded}" width="660" style="border-radius:100px">
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
                <br>
                <br>
                <h2>프로젝트 소개</h2>
                <h3 style='font-size:18px; color: #2C2C2C;'>
                    저희 프로젝트는 드론이 촬영한 항공 영상을 딥러닝으로 분석하여,
                    실종자를 탐지하는 것을 목표로 합니다.<br><br>
                </h3>

                <h3>스마트폰,웹캠 등에서 실시간 영상을 전송받아 분석<br><br>
                이미지를 업로드 받아 분석<br><br>
                이미 녹화된 영상을 업로드 받아 분석<br></h3>        
            """,
            unsafe_allow_html=True
        )

        
