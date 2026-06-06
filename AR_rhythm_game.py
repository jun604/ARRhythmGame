import cv2 as cv
import numpy as np
import random
import datetime
import librosa
import time
import pygame
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# [해상도 및 가상 게임보드 크기 설정]
# ==========================================
RESOLUTION = (640, 480)
SCAN_BOARD_SIZE = (480, 480)    # 실제 카메라 바닥에서 얻어낼 크기 (우측 영역)
VIRTUAL_BOARD_SIZE = (480, 720) # 노트가 흘러내려오는 원본 긴 보드 크기
NOTE_SPEED = 500 # 노트가 떨어지는 속도 (픽셀/초)
JUDGE_LINE_Y = VIRTUAL_BOARD_SIZE[1] - 80  # 720 - 80 = 640
lead_time = JUDGE_LINE_Y / NOTE_SPEED # 노트가 생성되어 판정선에 도달하기까지 걸리는 시간 (초)
win_name = "AR Rhythm Game"
fps = 30.0  # 일반적인 웹캠 FPS
fourcc = cv.VideoWriter_fourcc(*'XVID') # AVI 저장을 위한 코덱

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

def init_settings():
    pass

def make_recorder(cap, recorder_name):
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    recorder_name = f"{recorder_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.avi"
    return cv.VideoWriter(recorder_name, fourcc, fps, (width, height))

def record_video(frame, recorder):
    if recorder is not None:
        recorder.write(frame)

def put_korean_text(frame, text, org, fontScale, color, thickness=1):
    """
    기존 변수명을 완벽히 유지하면서 cv.putText와 인자를 호환하도록 한 함수
    """
    # cv.putText 인자를 기존 내부 로직 변수명(position, font_size)으로 변환 및 보정
    # 1. fontScale과 thickness를 기반으로 font_size(Pixel) 계산
    font_size = int(fontScale * 25) + (thickness * 2)
    
    # 2. cv.putText의 좌측 하단(Left-Bottom) 기준 좌표를 PIL의 좌측 상단(Left-Top) 기준으로 변환
    position = (org[0], org[1] - font_size)

    # 1. OpenCV 이미지(BGR)를 PIL 이미지(RGB)로 변환
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)
    
    # 2. PIL ImageDraw 객체 생성 및 폰트 지정
    draw = ImageDraw.Draw(pil_frame)
    
    # 윈도우 기본 폰트인 맑은 고딕(malgun.ttf) 사용 (경로는 환경에 맞게 지정)
    # 리눅스/맥의 경우 나눔고딕이나 AppleGothic 등 시스템 내 폰트 경로 입력
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except IOError:
        font = ImageFont.load_default() # 폰트가 없을 경우 기본 폰트 대체
        
    # 3. 한글 텍스트 그리기 (색상은 RGB 순서임에 주의)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    
    # 4. PIL 이미지를 다시 OpenCV 이미지(BGR)로 변환
    result_frame = cv.cvtColor(np.array(pil_frame), cv.COLOR_RGB2BGR)
    
    # 원본 frame 픽셀 데이터 자체를 업데이트하여 리턴 (cv.putText와 동일한 파괴적 동작 방식 매핑)
    frame[:] = result_frame
    return result_frame

def is_gesture_changed(prev_gesture, current_gesture):
    """
    ✊(주먹)와 ✋(손) 간의 제스처 전환이 일어났는지 판단하는 공통 함수
    """
    VALID_GESTURES = ["주먹", "손"]
    return (prev_gesture in VALID_GESTURES and 
            current_gesture in VALID_GESTURES and 
            prev_gesture != current_gesture)

def select_picture(frame, prev_hand_type=None, btn_x1=520, btn_x2=640, win_name="Select Picture"):
    """
    사용자가 스페이스바를 누르거나 손의 제스처를 바꿀 때 조건이 충족되어 사진을 확정합니다.
    """
    btn_y1, btn_y2 = 180, 300

    detected_hands = analyze_hand_gesture_mp(frame)
    
    gesture_changed = False
    is_hand_in_zone = False
    best_current_hand_type = None

    for (cx, cy, current_hand_type) in detected_hands:
        if cx is not None and current_hand_type in ["주먹", "손"]:
            if btn_x1 <= cx <= btn_x2 and btn_y1 <= cy <= btn_y2:
                is_hand_in_zone = True
                best_current_hand_type = current_hand_type
                
                color = (0, 255, 0) if current_hand_type == "손" else (0, 0, 255)
                cv.circle(frame, (cx, cy), 10, color, -1)
                put_korean_text(frame, current_hand_type, (cx - 30, cy - 20), 0.6, (0, 0, 0), 7)
                put_korean_text(frame, current_hand_type, (cx - 30, cy - 20), 0.6, color, 3)

                if is_gesture_changed(prev_hand_type, current_hand_type):
                    gesture_changed = True
                    print(f"✊✋ 버튼 영역 내 제스처 변경 감지! ({prev_hand_type} -> {current_hand_type})")
            else:
                cv.circle(frame, (cx, cy), 6, (255, 255, 0), -1)

    if is_hand_in_zone:
        prev_hand_type = best_current_hand_type
    else:
        prev_hand_type = None

    put_korean_text(frame, "박스 안에서 손 동작을 바꾸거나 [Spacebar]를 눌러주세요", (30, 50), 0.6, (0, 255, 255), 2)
    
    box_color = (0, 255, 0) if is_hand_in_zone else (255, 0, 0)
    box_thickness = 4 if is_hand_in_zone else 2
    cv.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), box_color, box_thickness)
    put_korean_text(frame, "시작 지점", (btn_x1, btn_y1 - 10), 0.5, box_color, 1)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord(' ') or gesture_changed:
        prev_hand_type = None
        return frame, prev_hand_type, True

    elif key == ord('q'):
        cv.destroyWindow(win_name)
        exit()
    else:
        return frame, prev_hand_type, False

def find_flat(cap, ref_img, out_recorder=None, scan_board_size=SCAN_BOARD_SIZE):
    if len(ref_img.shape) == 3:
        ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img.copy()

    orb = cv.ORB_create(nfeatures=2000)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    h_ref, w_ref = ref_gray.shape[:2]
    ref_pts = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
    #win_name = "AR Camera (Floor Scan Mode)"
    prev_hand_type = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            return None

        new_frame, prev_hand_type, gesture_changed = select_picture(frame, prev_hand_type, win_name=win_name)
        if new_frame is not None:
            frame = new_frame

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)

        if des_frame is not None and len(des_frame) > 10:
            matches = bf.match(des_ref, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:50]

            if len(good_matches) > 10:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                if H is not None:
                    dynamic_src_points = cv.perspectiveTransform(ref_pts, H)
                    pts = np.int32(dynamic_src_points)
                    
                    cv.polylines(frame, [pts], True, (0, 255, 0), 3)
                    
                    arrow_src_pts = np.float32([
                        [240, -150],  
                        [240, 100]   
                    ]).reshape(-1, 1, 2)
                    
                    arrow_dst_pts = cv.perspectiveTransform(arrow_src_pts, H)
                    
                    p_start = (int(arrow_dst_pts[0][0][0]), int(arrow_dst_pts[0][0][1]))
                    p_end = (int(arrow_dst_pts[1][0][0]), int(arrow_dst_pts[1][0][1]))
                    
                    cv.arrowedLine(frame, p_start, p_end, (0, 255, 255), 5, tipLength=0.3)

                    put_korean_text(frame, "일치 확인! 제스처를 바꿔 시작하세요", (30, 80), 0.6, (0, 255, 0), 2)

                    src_points = dynamic_src_points.reshape(4, 2)
                    
                    dst_points = np.float32([
                        [0, 0], 
                        [scan_board_size[0], 0], 
                        [scan_board_size[0], scan_board_size[1]], 
                        [0, scan_board_size[1]]
                    ])
                    M = cv.getPerspectiveTransform(src_points, dst_points)
                    if M is not None and gesture_changed:
                        print("-> 바닥 좌표(480x480 대응) 추출 성공! 게임을 시작합니다.")
                        
                        return M
            else:
                put_korean_text(frame, "바닥 마커를 스캔 중...", (30, 80), 0.6, (0, 0, 255), 2)
        record_video(frame, out_recorder)
        cv.imshow(win_name, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return None

def analyze_hand_gesture_mp(frame):
    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)
    hands_info = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            cx = int(hand_landmarks.landmark[9].x * w)
            cy = int(hand_landmarks.landmark[9].y * h)
            
            folded_fingers = 0
            tip_ids = [8, 12, 16, 20]
            pip_ids = [6, 10, 14, 18]
            
            for tip, pip in zip(tip_ids, pip_ids):
                if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
                    folded_fingers += 1
            
            hand_type = "주먹" if folded_fingers >= 3 else "손"
            hands_info.append((cx, cy, hand_type))
            
    return hands_info

def make_notes(AUDIO_FILE):
    print(f"🎵 [librosa 오디오 분석 중] 채보 자동 생성 중...")
    try:
        y, sr = librosa.load(AUDIO_FILE, sr=None)
        _, y_percussive = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        
        onset_frames = librosa.util.peak_pick(
            onset_env, pre_max=4, post_max=4, pre_avg=4, post_avg=6, delta=0.4, wait=12
        )
        note_timestamps = librosa.frames_to_time(onset_frames, sr=sr)
        
        generated_notes = []
        for ts in note_timestamps:
            if ts > 0.5:
                lane = random.randint(0, 3)
                generated_notes.append([ts, lane, False])
        return generated_notes
    except Exception as e:
        print(f"❌ 기본 데모 채보로 대체합니다. ({e})")
        return [[i * 0.5, random.randint(0, 3), False] for i in range(2, 100)]

def game_settings(cap, out_recorder=None):
    prev_hand_type = None
    gesture_changed = False
    #win_name = "AR Camera (Initialization)"
    last_clean_frame = None

    while not gesture_changed:
        ret, frame = cap.read()
        if not ret:
            exit()
        last_clean_frame = frame.copy()
        frame, prev_hand_type, gesture_changed = select_picture(frame, prev_hand_type, 0, 140, win_name=win_name)
        guide_pts = np.array([[160, 0], [640, 0], [640, 480], [160, 480]], dtype=np.int32)
        cv.polylines(frame, [guide_pts], True, (255, 255, 0), 2)
        put_korean_text(frame, "이 480x480 박스 안에 마커를 정렬해주세요", (170, 30), 0.5, (255, 255, 0), 1)
        record_video(frame, out_recorder)
        cv.imshow(win_name, frame)

    if last_clean_frame is not None:
        ref_cropped_img = last_clean_frame[0:480, 160:640]
    else:
        print("프레임을 캡처하지 못했습니다.")
        exit()

    M = find_flat(cap, ref_cropped_img, out_recorder, SCAN_BOARD_SIZE)
    if M is None:
        print("바닥 스캔 실패")
        exit()

    src_project_pts = np.float32([
        [0, VIRTUAL_BOARD_SIZE[1] - SCAN_BOARD_SIZE[1]],                  
        [SCAN_BOARD_SIZE[0], VIRTUAL_BOARD_SIZE[1] - SCAN_BOARD_SIZE[1]], 
        [SCAN_BOARD_SIZE[0], VIRTUAL_BOARD_SIZE[1]],                      
        [0, VIRTUAL_BOARD_SIZE[1]]                                        
    ])

    dst_project_pts = np.float32([
        [0, 0],
        [SCAN_BOARD_SIZE[0], 0],
        [SCAN_BOARD_SIZE[0], SCAN_BOARD_SIZE[1]],
        [0, SCAN_BOARD_SIZE[1]]
    ])

    M_virtual_to_scan = cv.getPerspectiveTransform(src_project_pts, dst_project_pts)
    M_inv = np.linalg.inv(M)
    M_final_overlay = np.dot(M_inv, M_virtual_to_scan)
    return M, M_inv, M_final_overlay

def play_game(cap, out_recorder, music_file=None, M=None, M_inv=None, M_final_overlay=None):
    while True: # 전체 복귀 루프 구현
        lane_x = [60, 180, 300, 420]

        # -------------------------------------------------------------
        # [신규 기능] 게임 시작 전 "방구석 리듬세상" 타이틀 2초 연출
        # -------------------------------------------------------------
        title_start_time = time.time()
        title_text = "방구석 리듬세상" # 영문 매핑 혹은 윈도우 환경 한글 호환 텍스트
        
        while time.time() - title_start_time < 2.0:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 중앙 정렬을 위한 타이틀 텍스트 크기 계산
            t_size = cv.getTextSize(title_text, cv.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
            tx = (RESOLUTION[0] - t_size[0]) // 2
            ty = (RESOLUTION[1] + t_size[1]) // 2
            
            # 뒷배경 반투명 느낌의 블랙 박스 및 테두리 연출
            cv.rectangle(frame, (tx - 30, ty - t_size[1] - 25), (tx + t_size[0] + 30, ty + 25), (15, 15, 15), -1)
            cv.rectangle(frame, (tx - 30, ty - t_size[1] - 25), (tx + t_size[0] + 30, ty + 25), (0, 215, 255), 3) # 황금빛 테두리
            
            # 타이틀 메인 텍스트 출력
            put_korean_text(frame, title_text, (tx, ty), 1.3, (0, 255, 255), 3) # 밝은 청록색 텍스트
            record_video(frame, out_recorder) # 타이틀 화면도 녹화
            cv.imshow(win_name, frame)
            cv.waitKey(1)

        if music_file is None and M is not None and M_inv is not None and M_final_overlay is not None:
            choice = ask_re_scan(cap, out_recorder)
            if choice == "재스캔":
                print("-> 바닥을 다시 스캔합니다.")
                M, M_inv, M_final_overlay = None, None, None

        if M is None or M_inv is None or M_final_overlay is None:
            M, M_inv, M_final_overlay = game_settings(cap, out_recorder)
        else:
            print("-> 이전 설정 유지, 바로 게임을 시작합니다.")
        if music_file is None:
            music_file = select_music_file(cap, out_recorder, M, M_inv, M_final_overlay)

        # -------------------------------------------------------------
        # 요구사항 1: 채보 생성 전 "리듬 노트 생성중" 화면 출력
        # -------------------------------------------------------------
        for _ in range(30): # 약 0.3초간 해당 화면 유지
            ret, frame = cap.read()
            if ret:
                text = "리듬 노트 생성중..."
                text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                box_x1 = (RESOLUTION[0] - text_size[0]) // 2 - 20
                box_y1 = (RESOLUTION[1] - text_size[1]) // 2 - 20
                box_x2 = (RESOLUTION[0] + text_size[0]) // 2 + 20
                box_y2 = (RESOLUTION[1] + text_size[1]) // 2 + 20
                
                cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1) 
                cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2) 
                put_korean_text(frame, text, ((RESOLUTION[0] - text_size[0]) // 2, (RESOLUTION[1] + text_size[1]) // 2),
                               0.8, (0, 255, 255), 2)
                record_video(frame, out_recorder)
                cv.imshow(win_name, frame)
                cv.waitKey(10)

        score = 0
        hp_bar = 1.0  # 초기 체력 가득 참 (1.0)
        AUDIO_FILE = music_file
        pygame.mixer.music.load(AUDIO_FILE)

        game_notes = make_notes(AUDIO_FILE)
        active_notes = []
        active_effects = []

        play_recorder = make_recorder(cap, "Play_video") # 게임 플레이 녹화 시작

        print("\n=== [2단계: 게임 시작] ===")
        for count in range(3, 0, -1):
                start_t = time.time()
                while time.time() - start_t < 1.0:
                    ret, f_count = cap.read()
                    if ret:
                        put_korean_text(f_count, f"시작까지 {count}초", (160, 240), 1.2, (0, 255, 255), 3)
                        record_video(f_count, out_recorder)
                        record_video(f_count, play_recorder)
                        cv.imshow(win_name, f_count)
                        cv.waitKey(1)

        pygame.mixer.music.play(0)
        game_over = False
        prev_hand_type = None

        # [메인 게임 플레이 루프]
        while not game_over:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 노래가 끝났거나 체력이 다 떨어진 경우 판정 탈출
            if not pygame.mixer.music.get_busy() or hp_bar <= 0.0:
                game_over = True
                break
                
            elapsed_time = pygame.mixer.music.get_pos() / 1000.0
            current_time = time.time()
            
            virtual_board = np.zeros((VIRTUAL_BOARD_SIZE[1], VIRTUAL_BOARD_SIZE[0], 3), dtype=np.uint8)
            
            for note in game_notes:
                target_time, lane, is_spawned = note
                if not is_spawned and elapsed_time >= (target_time - lead_time):
                    active_notes.append([lane_x[lane], 0, lane, target_time])
                    note[2] = True 
                
            cv.line(virtual_board, (0, JUDGE_LINE_Y), (VIRTUAL_BOARD_SIZE[0], JUDGE_LINE_Y), (255, 0, 0), 5)
            
            for note in active_notes[:]:
                time_to_target = note[3] - elapsed_time
                note[1] = int(JUDGE_LINE_Y - (time_to_target * NOTE_SPEED))
                
                if note[1] > VIRTUAL_BOARD_SIZE[1] - 10:
                    hp_bar -= 0.02  # Miss 발생 시 체력 0.02 감소 (50번 누적 시 종료되는 규격 동일 유지)
                    if hp_bar < 0.0: hp_bar = 0.0
                    
                    active_effects.append({
                        "text": "miss",
                        "color": (128, 128, 128),
                        "expire_time": current_time + 0.5,
                        "v_x": note[0] - 35,
                        "v_y": VIRTUAL_BOARD_SIZE[1] - 30 
                    })
                    active_notes.remove(note)
                    continue
                    
                cv.rectangle(virtual_board, (note[0]-50, note[1]-15), (note[0]+50, note[1]+15), (0, 0, 255), -1)

            for fx in active_effects[:]:
                if current_time > fx["expire_time"]:
                    active_effects.remove(fx)
                else:
                    cv.putText(virtual_board, fx["text"].upper(), (fx["v_x"], fx["v_y"]), 
                            cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 9)  
                    cv.putText(virtual_board, fx["text"].upper(), (fx["v_x"], fx["v_y"]), 
                            cv.FONT_HERSHEY_DUPLEX, 0.8, fx["color"], 3)  

            warped_game_overlay = cv.warpPerspective(virtual_board, M_final_overlay, RESOLUTION)
            overlay_gray = cv.cvtColor(warped_game_overlay, cv.COLOR_BGR2GRAY)
            _, mask_inv = cv.threshold(overlay_gray, 1, 255, cv.THRESH_BINARY_INV)
            
            background = cv.bitwise_and(frame, frame, mask=mask_inv)
            ar_frame = cv.add(background, warped_game_overlay)

            detected_hands = analyze_hand_gesture_mp(frame)
            current_hand_types = []
            
            for (cx, cy, current_hand_type) in detected_hands:
                current_hand_types.append(current_hand_type)
                
                hand_point = np.array([[[cx, cy]]], dtype=np.float32)
                transformed_hand = cv.perspectiveTransform(hand_point, M)
                hx, hy = transformed_hand[0][0][0], transformed_hand[0][0][1]
                
                scan_judge_y = SCAN_BOARD_SIZE[1] - 80 
                
                if abs(hy - scan_judge_y) <= 110:
                    corrected_hand = cv.perspectiveTransform(np.array([[[hx, scan_judge_y]]], dtype=np.float32), M_inv)
                    cx_draw, cy_draw = int(corrected_hand[0][0][0]), int(corrected_hand[0][0][1])
                else:
                    cx_draw, cy_draw = cx, cy

                color = (0, 255, 0) if current_hand_type == "손" else (0, 0, 255)
                cv.circle(ar_frame, (cx_draw, cy_draw), 10, color, -1)
                put_korean_text(ar_frame, current_hand_type, (cx_draw - 30, cy_draw - 20), 0.6, (0, 0, 0), 7)
                put_korean_text(ar_frame, current_hand_type, (cx_draw - 30, cy_draw - 20), 0.6, color, 3)

                if is_gesture_changed(prev_hand_type, current_hand_type) and 0 <= hx < SCAN_BOARD_SIZE[0]:
                    for note in active_notes[:]:
                        if (abs(hx - note[0]) < 50) and (abs(elapsed_time - note[3]) < 0.15):
                            score += 100
                            hp_bar += 0.002  # Hit 성공 시 체력 0.1 회복
                            if hp_bar > 1.0: hp_bar = 1.0  # 최대 체력 제한
                            
                            active_effects.append({
                                "text": "hit",
                                "color": (0, 0, 255), 
                                "expire_time": current_time + 0.5,
                                "v_x": note[0] - 25,
                                "v_y": JUDGE_LINE_Y - 20 
                            })
                            active_notes.remove(note)

            if current_hand_types:
                prev_hand_type = current_hand_types[0]
            else:
                prev_hand_type = None

            # -------------------------------------------------------------
            # [UI 컴포넌트 추가] SCORE 표시 및 실시간 체력 게이지 바 그리기
            # -------------------------------------------------------------
            cv.putText(ar_frame, f"SCORE: {score}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 게이지 바 위치 세팅 (우측 상단 배치)
            bar_x1, bar_y1 = 400, 20
            bar_x2, bar_y2 = 620, 40
            bar_width = bar_x2 - bar_x1
            current_bar_w = int(bar_width * hp_bar)
            
            # 위험 상태(체력 30% 이하)일 때는 빨간색, 평소에는 초록색
            hp_color = (0, 0, 255) if hp_bar <= 0.3 else (0, 255, 0)
            
            # 배경 빈 바(Dark Gray)와 현재 체력 게이지 바 채우기
            cv.rectangle(ar_frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
            if current_bar_w > 0:
                cv.rectangle(ar_frame, (bar_x1, bar_y1), (bar_x1 + current_bar_w, bar_y2), hp_color, -1)
            cv.rectangle(ar_frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1) # 테두리 흰색
            cv.putText(ar_frame, "HP", (bar_x1 - 35, bar_y1 + 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            record_video(ar_frame, out_recorder)
            record_video(ar_frame, play_recorder)
            cv.imshow(win_name, ar_frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                pygame.mixer.music.stop()
                return

        # -------------------------------------------------------------
        # 점수 표시 전 "GAME OVER" / "GAME CLEAR" 3초 연출
        # -------------------------------------------------------------
        pygame.mixer.music.stop()
        
        # 조건 판별: 체력이 다 떨어져서 끝났으면 오버, 아니면 클리어
        if hp_bar <= 0.0:
            result_text = "GAME OVER"
            text_color = (0, 0, 255)  # 빨간색
        else:
            result_text = "GAME CLEAR"
            text_color = (0, 255, 0)  # 초록색

        end_display_start = time.time()
        while time.time() - end_display_start < 3.0:
            ret, frame = cap.read()
            if not ret:
                break
            
            t_size = cv.getTextSize(result_text, cv.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            tx = (RESOLUTION[0] - t_size[0]) // 2
            ty = (RESOLUTION[1] + t_size[1]) // 2
            
            cv.rectangle(frame, (tx - 20, ty - t_size[1] - 20), (tx + t_size[0] + 20, ty + 20), (0, 0, 0), -1)
            cv.rectangle(frame, (tx - 20, ty - t_size[1] - 20), (tx + t_size[0] + 20, ty + 20), text_color, 2)
            
            cv.putText(frame, result_text, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 4, cv.LINE_AA)
            record_video(frame, out_recorder)
            record_video(frame, play_recorder)
            cv.imshow(win_name, frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # -------------------------------------------------------------
        # 요구사항 2~4: Game Over 결과창 및 동일 규격 메뉴 인터랙션 설계
        # -------------------------------------------------------------
        menu_action = None
        result_prev_hand = None

        btn_w, btn_h = 140, 50
        gap = 50
        
        start_x = (RESOLUTION[0] - (btn_w * 3 + gap * 2)) // 2  
        
        btn1_x1, btn1_x2 = start_x, start_x + btn_w
        btn2_x1, btn2_x2 = btn1_x2 + gap, btn1_x2 + gap + btn_w
        btn3_x1, btn3_x2 = btn2_x2 + gap, btn2_x2 + gap + btn_w
        
        btn_y1, btn_y2 = 300, 300 + btn_h

        while menu_action is None:
            ret, frame = cap.read()
            if not ret:
                break

            score_box_x1, score_box_y1 = 170, 120
            score_box_x2, score_box_y2 = 470, 220
            cv.rectangle(frame, (score_box_x1, score_box_y1), (score_box_x2, score_box_y2), (20, 20, 20), -1) 
            cv.rectangle(frame, (score_box_x1, score_box_y1), (score_box_x2, score_box_y2), (0, 215, 255), 3) 
            
            score_txt = f"FINAL SCORE: {score}"
            txt_w, txt_h = cv.getTextSize(score_txt, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv.putText(frame, score_txt, (170 + (300 - txt_w)//2, 120 + (100 + txt_h)//2), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            detected_hands = analyze_hand_gesture_mp(frame)
            active_btn_idx = 0 
            current_hand_type_found = None
            h_cx, h_cy = None, None

            for (cx, cy, current_hand_type) in detected_hands:
                if cx is not None:
                    h_cx, h_cy = cx, cy
                    current_hand_type_found = current_hand_type
                    if btn_y1 <= cy <= btn_y2:
                        if btn1_x1 <= cx <= btn1_x2: active_btn_idx = 1
                        elif btn2_x1 <= cx <= btn2_x2: active_btn_idx = 2
                        elif btn3_x1 <= cx <= btn3_x2: active_btn_idx = 3

            colors = [(255, 0, 0), (255, 0, 0), (255, 0, 0)] 
            if active_btn_idx > 0:
                colors[active_btn_idx - 1] = (0, 255, 0) 

            cv.rectangle(frame, (btn1_x1, btn_y1), (btn1_x2, btn_y2), colors[0], 2 if active_btn_idx != 1 else 4)
            put_korean_text(frame, "메인으로 돌아가기", (btn1_x1 + 10, btn_y1 + 30), 0.5, (255, 255, 255), 1)

            cv.rectangle(frame, (btn2_x1, btn_y1), (btn2_x2, btn_y2), colors[1], 2 if active_btn_idx != 2 else 4)
            put_korean_text(frame, "다시 시작", (btn2_x1 + 35, btn_y1 + 30), 0.5, (255, 255, 255), 1)

            cv.rectangle(frame, (btn3_x1, btn_y1), (btn3_x2, btn_y2), colors[2], 2 if active_btn_idx != 3 else 4)
            put_korean_text(frame, "게임 종료", (btn3_x1 + 25, btn_y1 + 30), 0.5, (255, 255, 255), 1)

            if current_hand_type_found and h_cx is not None:
                dot_color = (0, 255, 0) if current_hand_type_found == "손" else (0, 0, 255)
                cv.circle(frame, (h_cx, h_cy), 8, dot_color, -1)
                put_korean_text(frame, current_hand_type_found, (h_cx - 20, h_cy - 15), 0.5, dot_color, 2)

                if is_gesture_changed(result_prev_hand, current_hand_type_found):
                    if active_btn_idx == 1:
                        menu_action = "MAIN"
                    elif active_btn_idx == 2:
                        menu_action = "RESTART"
                    elif active_btn_idx == 3:
                        menu_action = "EXIT"
                
                result_prev_hand = current_hand_type_found
            else:
                result_prev_hand = None
            record_video(frame, out_recorder)
            record_video(frame, play_recorder)
            cv.imshow(win_name, frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                menu_action = "EXIT"

        if menu_action == "MAIN":
            print("-> 처음 세팅 화면으로 복귀합니다.")
            music_file = None
            
        elif menu_action == "RESTART":
            print("-> 게임을 다시 시작합니다.")
            
        elif menu_action == "EXIT":
            print("-> 프로그램을 종료합니다.")
            break


def select_music_file(cap, out_recorder=None, M=None, M_inv=None, M_final_overlay=None):
    #win_name = "AR Rhythm Game Play Board (Camera View)"
    prev_hand_type = None
    selected_music = None

    if M is None or M_inv is None or M_final_overlay is None:
        M, M_inv, M_final_overlay = game_settings(cap, out_recorder)

    btn_w, btn_h = 160, 60
    gap = 60
    
    start_x = (RESOLUTION[0] - (btn_w * 2 + gap)) // 2  
    
    btn1_x1, btn1_x2 = start_x, start_x + btn_w
    btn2_x1, btn2_x2 = btn1_x2 + gap, btn1_x2 + gap + btn_w
    
    btn_y1, btn_y2 = 260, 260 + btn_h

    print("\n=== [음악 선택 메뉴 진입] ===")

    while selected_music is None:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            return "classic.mp3"  
        
        title_box_x1, title_box_y1 = 150, 100
        title_box_x2, title_box_y2 = 490, 190
        cv.rectangle(frame, (title_box_x1, title_box_y1), (title_box_x2, title_box_y2), (20, 20, 20), -1) 
        cv.rectangle(frame, (title_box_x1, title_box_y1), (title_box_x2, title_box_y2), (255, 215, 0), 3) 
        
        title_txt = "SELECT MUSIC"
        txt_w, txt_h = cv.getTextSize(title_txt, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        put_korean_text(frame, title_txt, (title_box_x1 + (340 - txt_w)//2, title_box_y1 + (90 + txt_h)//2), 0.8, (255, 255, 255), 2)

        detected_hands = analyze_hand_gesture_mp(frame)
        active_btn_idx = 0 
        current_hand_type_found = None
        h_cx, h_cy = None, None

        for (cx, cy, current_hand_type) in detected_hands:
            if cx is not None:
                h_cx, h_cy = cx, cy
                current_hand_type_found = current_hand_type
                if btn_y1 <= cy <= btn_y2:
                    if btn1_x1 <= cx <= btn1_x2: 
                        active_btn_idx = 1
                    elif btn2_x1 <= cx <= btn2_x2: 
                        active_btn_idx = 2

        colors = [(255, 0, 0), (255, 0, 0)]  
        if active_btn_idx > 0:
            colors[active_btn_idx - 1] = (0, 255, 0)  

        cv.rectangle(frame, (btn1_x1, btn_y1), (btn1_x2, btn_y2), colors[0], 2 if active_btn_idx != 1 else 4)
        pop_w = cv.getTextSize("pop", cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv.putText(frame, "pop", (btn1_x1 + (btn_w - pop_w)//2, btn_y1 + 38), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv.rectangle(frame, (btn2_x1, btn_y1), (btn2_x2, btn_y2), colors[1], 2 if active_btn_idx != 2 else 4)
        classic_w = cv.getTextSize("classic", cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv.putText(frame, "classic", (btn2_x1 + (btn_w - classic_w)//2, btn_y1 + 38), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if current_hand_type_found and h_cx is not None:
            dot_color = (0, 255, 0) if current_hand_type_found == "손" else (0, 0, 255)
            cv.circle(frame, (h_cx, h_cy), 8, dot_color, -1)
            put_korean_text(frame, current_hand_type_found, (h_cx - 20, h_cy - 15), 0.5, dot_color, 2)

            if is_gesture_changed(prev_hand_type, current_hand_type_found):
                if active_btn_idx == 1:
                    selected_music = "pop.mp3"
                elif active_btn_idx == 2:
                    selected_music = "classic.mp3"
            
            prev_hand_type = current_hand_type_found
        else:
            prev_hand_type = None

        record_video(frame, out_recorder)
        cv.imshow(win_name, frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            return "classic.mp3"  

    print(f"-> 선택된 음악: {selected_music}")
    return selected_music

def ask_re_scan(cap, out_recorder=None):
    """
    이전 바닥 스캔 데이터가 존재할 때, 기존 설정을 유지할지 새로 스캔할지 
    사용자의 제스처 입력을 받아 결정하는 UI 메뉴입니다.
    """
    #win_name = "AR Rhythm Game Play Board (Camera View)"
    prev_hand_type = None
    choice = None

    btn_w, btn_h = 160, 60
    gap = 60
    
    start_x = (RESOLUTION[0] - (btn_w * 2 + gap)) // 2  
    btn1_x1, btn1_x2 = start_x, start_x + btn_w
    btn2_x1, btn2_x2 = btn1_x2 + gap, btn1_x2 + gap + btn_w
    btn_y1, btn_y2 = 260, 260 + btn_h

    print("\n=== [바닥 스캔 재사용 선택 메뉴 진입] ===")

    while choice is None:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            return "유지"  
        
        title_box_x1, title_box_y1 = 120, 100
        title_box_x2, title_box_y2 = 520, 190
        cv.rectangle(frame, (title_box_x1, title_box_y1), (title_box_x2, title_box_y2), (20, 20, 20), -1) 
        cv.rectangle(frame, (title_box_x1, title_box_y1), (title_box_x2, title_box_y2), (255, 215, 0), 3) 
        
        title_txt = "USE PREVIOUS SCAN?"
        txt_w, txt_h = cv.getTextSize(title_txt, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        put_korean_text(frame, title_txt, (title_box_x1 + ((520 - 120) - txt_w)//2, title_box_y1 + (90 + txt_h)//2), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        detected_hands = analyze_hand_gesture_mp(frame)
        active_btn_idx = 0 
        current_hand_type_found = None
        h_cx, h_cy = None, None

        for (cx, cy, current_hand_type) in detected_hands:
            if cx is not None:
                h_cx, h_cy = cx, cy
                current_hand_type_found = current_hand_type
                if btn_y1 <= cy <= btn_y2:
                    if btn1_x1 <= cx <= btn1_x2: 
                        active_btn_idx = 1
                    elif btn2_x1 <= cx <= btn2_x2: 
                        active_btn_idx = 2

        colors = [(255, 0, 0), (255, 0, 0)]  
        if active_btn_idx > 0:
            colors[active_btn_idx - 1] = (0, 255, 0)  

        cv.rectangle(frame, (btn1_x1, btn_y1), (btn1_x2, btn_y2), colors[0], 2 if active_btn_idx != 1 else 4)
        keep_w = cv.getTextSize("유지", cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        put_korean_text(frame, "유지", (btn1_x1 + (btn_w - keep_w)//2, btn_y1 + 38), 0.6, (255, 255, 255), 2)

        cv.rectangle(frame, (btn2_x1, btn_y1), (btn2_x2, btn_y2), colors[1], 2 if active_btn_idx != 2 else 4)
        rescan_w = cv.getTextSize("재스캔", cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        put_korean_text(frame, "재스캔", (btn2_x1 + (btn_w - rescan_w)//2, btn_y1 + 38), 0.6, (255, 255, 255), 2)

        if current_hand_type_found and h_cx is not None:
            dot_color = (0, 255, 0) if current_hand_type_found == "손" else (0, 0, 255)
            cv.circle(frame, (h_cx, h_cy), 8, dot_color, -1)
            put_korean_text(frame, current_hand_type_found, (h_cx - 20, h_cy - 15), 0.5, dot_color, 2)

            if is_gesture_changed(prev_hand_type, current_hand_type_found):
                if active_btn_idx == 1:
                    choice = "유지"
                elif active_btn_idx == 2:
                    choice = "재스캔"
            
            prev_hand_type = current_hand_type_found
        else:
            prev_hand_type = None

        record_video(frame, out_recorder)
        cv.imshow(win_name, frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            return "유지"

    print(f"-> 사용자의 선택: {choice}")
    return choice


# -------------------------------------------------------------------------
# 설정 및 초기화
# -------------------------------------------------------------------------

pygame.mixer.init()
cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

out_recorder = make_recorder(cap, "Ar_rhythm_game")
play_game(cap, out_recorder=out_recorder)

cap.release()
cv.destroyAllWindows()