import cv2 as cv
import numpy as np
import random
import datetime
import librosa
import time
import pygame
import mediapipe as mp

# ==========================================
# [해상도 및 가상 게임보드 크기 설정]
# ==========================================
RESOLUTION = (640, 480)
SCAN_BOARD_SIZE = (480, 480)    # 실제 카메라 바닥에서 얻어낼 크기 (우측 영역)
VIRTUAL_BOARD_SIZE = (480, 720) # 노트가 흘러내려오는 원본 긴 보드 크기
NOTE_SPEED = 500 # 노트가 떨어지는 속도 (픽셀/초)
JUDGE_LINE_Y = VIRTUAL_BOARD_SIZE[1] - 80  # 720 - 80 = 640
lead_time = JUDGE_LINE_Y / NOTE_SPEED # 노트가 생성되어 판정선에 도달하기까지 걸리는 시간 (초)

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

def init_settings():
    pass

def record_video(frame, recorder):
    if recorder is not None:
        recorder.write(frame)

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
        if cx is not None and current_hand_type in ["FIST", "HAND"]:
            if btn_x1 <= cx <= btn_x2 and btn_y1 <= cy <= btn_y2:
                is_hand_in_zone = True
                best_current_hand_type = current_hand_type
                
                color = (0, 255, 0) if current_hand_type == "HAND" else (0, 0, 255)
                cv.circle(frame, (cx, cy), 10, color, -1)
                cv.putText(frame, current_hand_type, (cx - 30, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 7)
                cv.putText(frame, current_hand_type, (cx - 30, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
                
                if prev_hand_type in ["FIST", "HAND"] and prev_hand_type != current_hand_type:
                    gesture_changed = True
                    print(f"✊✋ 버튼 영역 내 제스처 변경 감지! ({prev_hand_type} -> {current_hand_type})")
            else:
                cv.circle(frame, (cx, cy), 6, (255, 255, 0), -1)

    if is_hand_in_zone:
        prev_hand_type = best_current_hand_type
    else:
        prev_hand_type = None

    cv.putText(frame, "Change Gesture in BOX or Press [Spacebar]", (30, 50), 
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    box_color = (0, 255, 0) if is_hand_in_zone else (255, 0, 0)
    box_thickness = 4 if is_hand_in_zone else 2
    cv.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), box_color, box_thickness)
    cv.putText(frame, "START ZONE", (btn_x1 - 10, btn_y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord(' ') or gesture_changed:
        if win_name != "AR Camera (Floor Scan Mode)":
            cv.destroyWindow(win_name) 
        prev_hand_type = None
        return frame, prev_hand_type, True

    elif key == ord('q'):
        cv.destroyWindow(win_name)
        exit()
    else:
        return frame, prev_hand_type, False

def find_flat(cap, ref_img, scan_board_size=SCAN_BOARD_SIZE):
    if len(ref_img.shape) == 3:
        ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img.copy()

    orb = cv.ORB_create(nfeatures=2000)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    h_ref, w_ref = ref_gray.shape[:2]
    ref_pts = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
    win_name = "AR Camera (Floor Scan Mode)"
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

                    cv.putText(frame, "MATCH FOUND! Change gesture to Start", (30, 80), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
                        cv.destroyWindow(win_name)
                        return M
            else:
                cv.putText(frame, "Scanning floor marker...", (30, 80), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
            
            hand_type = "FIST" if folded_fingers >= 3 else "HAND"
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

def game_settings(cap):
    prev_hand_type = None
    gesture_changed = False
    win_name = "AR Camera (Initialization)"
    last_clean_frame = None

    while not gesture_changed:
        ret, frame = cap.read()
        if not ret:
            exit()
        last_clean_frame = frame.copy()
        frame, prev_hand_type, gesture_changed = select_picture(frame, prev_hand_type, 0, 140, win_name=win_name)
        guide_pts = np.array([[160, 0], [640, 0], [640, 480], [160, 480]], dtype=np.int32)
        cv.polylines(frame, [guide_pts], True, (255, 255, 0), 2)
        cv.putText(frame, "Align Marker inside this 480x480 BOX", (170, 30), 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv.imshow(win_name, frame)
    cv.destroyWindow(win_name)

    if last_clean_frame is not None:
        ref_cropped_img = last_clean_frame[0:480, 160:640]
    else:
        print("프레임을 캡처하지 못했습니다.")
        exit()

    M = find_flat(cap, ref_cropped_img, SCAN_BOARD_SIZE)
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

def play_game(cap, music_file="classic.mp3", M=None, M_inv=None, M_final_overlay=None):
    while True: # 전체 복귀 루프 구현
        lane_x = [60, 180, 300, 420]
        
        if M is None or M_inv is None or M_final_overlay is None:
            M, M_inv, M_final_overlay = game_settings(cap)
        else:
            print("-> 이전 설정 유지, 바로 게임을 시작합니다.")

        # -------------------------------------------------------------
        # 요구사항 1: 채보 생성 전 "리듬 노트 생성중" 화면 출력
        # -------------------------------------------------------------
        for _ in range(50): # 약 0.5초간 해당 화면 유지 (30프레임 * 100ms)
            ret, frame = cap.read()
            if ret:
                # 중앙 정렬을 위한 텍스트 배경 박스 및 문자 출력
                text = "Generating Rhythm Notes..."
                text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                box_x1 = (RESOLUTION[0] - text_size[0]) // 2 - 20
                box_y1 = (RESOLUTION[1] - text_size[1]) // 2 - 20
                box_x2 = (RESOLUTION[0] + text_size[0]) // 2 + 20
                box_y2 = (RESOLUTION[1] + text_size[1]) // 2 + 20
                
                cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1) # 검은색 꽉 찬 박스
                cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2) # 노란색 테두리
                cv.putText(frame, text, ((RESOLUTION[0] - text_size[0]) // 2, (RESOLUTION[1] + text_size[1]) // 2),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv.imshow("AR Rhythm Game Play Board (Camera View)", frame)
                cv.waitKey(10) # 10ms 대기하여 약 0.5초간 화면 갱신 유지

        score = 0
        miss_count = 0  # 미스 카운트 변수 추가
        AUDIO_FILE = music_file

        pygame.mixer.init()
        pygame.mixer.music.load(AUDIO_FILE)

        game_notes = make_notes(AUDIO_FILE)
        active_notes = []
        active_effects = []

        print("\n=== [2단계: 게임 시작] ===")
        for count in range(3, 0, -1):
                start_t = time.time()
                while time.time() - start_t < 1.0:
                    ret, f_count = cap.read()
                    if ret:
                        cv.putText(f_count, f"Starting in {count}...", (160, 240), 
                                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                        cv.imshow("AR Rhythm Game Play Board (Camera View)", f_count)
                        cv.waitKey(1)

        pygame.mixer.music.play(0)
        game_over = False
        prev_hand_type = None

        # [메인 게임 플레이 루프]
        while not game_over:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 노래가 끝났거나 미스가 50번 누적된 경우 판정 탈출
            if not pygame.mixer.music.get_busy() or miss_count >= 50:
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
                    miss_count += 1 # 미스 카운트 누적
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

                color = (0, 255, 0) if current_hand_type == "HAND" else (0, 0, 255)
                cv.circle(ar_frame, (cx_draw, cy_draw), 10, color, -1)
                cv.putText(ar_frame, current_hand_type, (cx_draw - 30, cy_draw - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 7)  
                cv.putText(ar_frame, current_hand_type, (cx_draw - 30, cy_draw - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
                
                gesture_changed = False
                if prev_hand_type is not None and prev_hand_type != current_hand_type:
                    gesture_changed = True
                    
                if gesture_changed and 0 <= hx < SCAN_BOARD_SIZE[0]:
                    for note in active_notes[:]:
                        if (abs(hx - note[0]) < 50) and (abs(elapsed_time - note[3]) < 0.15):
                            score += 100
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

            cv.putText(ar_frame, f"SCORE: {score}  MISS: {miss_count}/50", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv.imshow("AR Rhythm Game Play Board (Camera View)", ar_frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                pygame.mixer.music.stop()
                return

        # -------------------------------------------------------------
        # 요구사항 2~4: Game Over 결과창 및 동일 규격 메뉴 인터랙션 설계
        # -------------------------------------------------------------
        pygame.mixer.music.stop()
        menu_action = None
        result_prev_hand = None

        # 버튼 규격 설정 (동일 크기 가로 140, 세로 50, 간격 50)
        btn_w, btn_h = 140, 50
        gap = 50
        
        # 전체 가로 정렬을 위한 시작 X 좌표 계산 (화면 중앙 근처 배치)
        start_x = (RESOLUTION[0] - (btn_w * 3 + gap * 2)) // 2  # (640 - (420 + 100)) // 2 = 60
        
        btn1_x1, btn1_x2 = start_x, start_x + btn_w
        btn2_x1, btn2_x2 = btn1_x2 + gap, btn1_x2 + gap + btn_w
        btn3_x1, btn3_x2 = btn2_x2 + gap, btn2_x2 + gap + btn_w
        
        btn_y1, btn_y2 = 300, 300 + btn_h

        while menu_action is None:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 최종 점수 UI 박스 그리기 (화면 중앙 상단 부근)
            score_box_x1, score_box_y1 = 170, 120
            score_box_x2, score_box_y2 = 470, 220
            cv.rectangle(frame, (score_box_x1, score_box_y1), (score_box_x2, score_box_y2), (20, 20, 20), -1) # 어두운 회색배경
            cv.rectangle(frame, (score_box_x1, score_box_y1), (score_box_x2, score_box_y2), (0, 215, 255), 3) # 금색빛 테두리
            
            score_txt = f"FINAL SCORE: {score}"
            txt_w, txt_h = cv.getTextSize(score_txt, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv.putText(frame, score_txt, (170 + (300 - txt_w)//2, 120 + (100 + txt_h)//2), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 2. 손 제스처 실시간 트래킹 및 영역 충족 조건 감지
            detected_hands = analyze_hand_gesture_mp(frame)
            active_btn_idx = 0 # 1: 돌아가기, 2: 다시하기, 3: 종료
            current_hand_type_found = None
            h_cx, h_cy = None, None

            for (cx, cy, current_hand_type) in detected_hands:
                if cx is not None:
                    h_cx, h_cy = cx, cy
                    current_hand_type_found = current_hand_type
                    # 어떤 버튼 영역에 들어왔는지 트리거 검사
                    if btn_y1 <= cy <= btn_y2:
                        if btn1_x1 <= cx <= btn1_x2: active_btn_idx = 1
                        elif btn2_x1 <= cx <= btn2_x2: active_btn_idx = 2
                        elif btn3_x1 <= cx <= btn3_x2: active_btn_idx = 3

            # 3. 버튼 3개 렌더링 (동일 크기, 간격 50)
            colors = [(255, 0, 0), (255, 0, 0), (255, 0, 0)] # 디폴트 파란색/하늘색 계열
            if active_btn_idx > 0:
                colors[active_btn_idx - 1] = (0, 255, 0) # 진입 시 초록색 하이라이트

            # 버튼 1: 돌아가기
            cv.rectangle(frame, (btn1_x1, btn_y1), (btn1_x2, btn_y2), colors[0], 2 if active_btn_idx != 1 else 4)
            cv.putText(frame, "Back to Main", (btn1_x1 + 10, btn_y1 + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 버튼 2: 다시하기
            cv.rectangle(frame, (btn2_x1, btn_y1), (btn2_x2, btn_y2), colors[1], 2 if active_btn_idx != 2 else 4)
            cv.putText(frame, "Restart", (btn2_x1 + 35, btn_y1 + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 버튼 3: 종료
            cv.rectangle(frame, (btn3_x1, btn_y1), (btn3_x2, btn_y2), colors[2], 2 if active_btn_idx != 3 else 4)
            cv.putText(frame, "Exit Game", (btn3_x1 + 25, btn_y1 + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 손 위치 피드백 표시
            if current_hand_type_found and h_cx is not None:
                dot_color = (0, 255, 0) if current_hand_type_found == "HAND" else (0, 0, 255)
                cv.circle(frame, (h_cx, h_cy), 8, dot_color, -1)
                cv.putText(frame, current_hand_type_found, (h_cx - 20, h_cy - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 2)

                # 제스처 변경 이벤트 트리거 확인
                if result_prev_hand in ["FIST", "HAND"] and result_prev_hand != current_hand_type_found:
                    if active_btn_idx == 1:
                        menu_action = "MAIN"
                    elif active_btn_idx == 2:
                        menu_action = "RESTART"
                    elif active_btn_idx == 3:
                        menu_action = "EXIT"
                
                result_prev_hand = current_hand_type_found
            else:
                result_prev_hand = None

            cv.imshow("AR Rhythm Game Play Board (Camera View)", frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                menu_action = "EXIT"

        # 4. 선택된 메뉴 액션 처리 분기문
        if menu_action == "MAIN":
            print("-> 처음 세팅 화면으로 복귀합니다.")
            M, M_inv, M_final_overlay = None, None, None # 기존 세팅 초기화
            cv.destroyAllWindows()
            
        elif menu_action == "RESTART":
            # 3초 카운트다운 가시적 시각화 연출 후 재시작
            print("-> 게임을 다시 시작합니다.")
            # 바닥 행렬 정보 M 등은 유지한 채 플레이 루프만 즉시 재진입하기 위해 세팅 복귀 처리 생략 가능하나, 
            # game_settings부터 안전하게 순차 진행되도록 설계됨. (필요 시 가상 변수만 재초기화 가능)
            
        elif menu_action == "EXIT":
            print("-> 프로그램을 종료합니다.")
            break


# -------------------------------------------------------------------------
# 설정 및 초기화
# -------------------------------------------------------------------------
cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

play_game(cap)

cap.release()
cv.destroyAllWindows()