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

def play_game(cap, music_file=None, M=None, M_inv=None, M_final_overlay=None):
    while True: # 전체 복귀 루프 구현
        lane_x = [60, 180, 300, 420]
        
        if M is None or M_inv is None or M_final_overlay is None:
            M, M_inv, M_final_overlay = game_settings(cap)
        else:
            print("-> 이전 설정 유지, 바로 게임을 시작합니다.")
        if music_file is None:
            music_file = select_music_file(cap, M, M_inv, M_final_overlay)

        # -------------------------------------------------------------
        # 요구사항 1: 채보 생성 전 "리듬 노트 생성중" 화면 출력
        # -------------------------------------------------------------
        for _ in range(30): # 약 0.3초간 해당 화면 유지
            ret, frame = cap.read()
            if ret:
                text = "Generating Rhythm Notes..."
                text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                box_x1 = (RESOLUTION[0] - text_size[0]) // 2 - 20
                box_y1 = (RESOLUTION[1] - text_size[1]) // 2 - 20
                box_x2 = (RESOLUTION[0] + text_size[0]) // 2 + 20
                box_y2 = (RESOLUTION[1] + text_size[1]) // 2 + 20
                
                cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1) 
                cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2) 
                cv.putText(frame, text, ((RESOLUTION[0] - text_size[0]) // 2, (RESOLUTION[1] + text_size[1]) // 2),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv.imshow("AR Rhythm Game Play Board (Camera View)", frame)
                cv.waitKey(10)

        score = 0
        miss_count = 0  
        AUDIO_FILE = music_file
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
                    miss_count += 1 
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
        # [추가된 요구사항] 점수 표시 전 "GAME OVER" / "GAME CLEAR" 3초 연출
        # -------------------------------------------------------------
        pygame.mixer.music.stop()
        
        # 조건 판별: 미스가 50개 이상 쌓여서 끝났으면 오버, 아니면 클리어
        if miss_count >= 50:
            result_text = "GAME OVER"
            text_color = (0, 0, 255)  # 빨간색
        else:
            result_text = "GAME CLEAR"
            text_color = (0, 255, 0)  # 초록색

        # 3초 동안 루프 돌며 화면에 고정 출력
        end_display_start = time.time()
        while time.time() - end_display_start < 3.0:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 중앙 정렬을 위한 텍스트 크기 계산
            t_size = cv.getTextSize(result_text, cv.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            tx = (RESOLUTION[0] - t_size[0]) // 2
            ty = (RESOLUTION[1] + t_size[1]) // 2
            
            # 어두운 배경 박스 깔아주기 (가독성 확보)
            cv.rectangle(frame, (tx - 20, ty - t_size[1] - 20), (tx + t_size[0] + 20, ty + 20), (0, 0, 0), -1)
            cv.rectangle(frame, (tx - 20, ty - t_size[1] - 20), (tx + t_size[0] + 20, ty + 20), text_color, 2)
            
            # 텍스트 출력
            cv.putText(frame, result_text, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 4, cv.LINE_AA)
            cv.imshow("AR Rhythm Game Play Board (Camera View)", frame)
            
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
            cv.putText(frame, "Back to Main", (btn1_x1 + 10, btn_y1 + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv.rectangle(frame, (btn2_x1, btn_y1), (btn2_x2, btn_y2), colors[1], 2 if active_btn_idx != 2 else 4)
            cv.putText(frame, "Restart", (btn2_x1 + 35, btn_y1 + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv.rectangle(frame, (btn3_x1, btn_y1), (btn3_x2, btn_y2), colors[2], 2 if active_btn_idx != 3 else 4)
            cv.putText(frame, "Exit Game", (btn3_x1 + 25, btn_y1 + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if current_hand_type_found and h_cx is not None:
                dot_color = (0, 255, 0) if current_hand_type_found == "HAND" else (0, 0, 255)
                cv.circle(frame, (h_cx, h_cy), 8, dot_color, -1)
                cv.putText(frame, current_hand_type_found, (h_cx - 20, h_cy - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 2)

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

        if menu_action == "MAIN":
            print("-> 처음 세팅 화면으로 복귀합니다.")
            M, M_inv, M_final_overlay, music_file = None, None, None, None
            cv.destroyAllWindows()
            
        elif menu_action == "RESTART":
            print("-> 게임을 다시 시작합니다.")
            
        elif menu_action == "EXIT":
            print("-> 프로그램을 종료합니다.")
            break

def select_music_file(cap, M=None, M_inv=None, M_final_overlay=None):
    win_name = "AR Rhythm Game Play Board (Camera View)"
    prev_hand_type = None
    selected_music = None

    if M is None or M_inv is None or M_final_overlay is None:
        M, M_inv, M_final_overlay = game_settings(cap)

    # 버튼 크기 및 배치 설정 (기존 종료 UI 규격과 동일하게 세팅)
    btn_w, btn_h = 160, 60
    gap = 60
    
    # 두 개의 버튼을 화면 중앙 정렬하기 위한 계산
    start_x = (RESOLUTION[0] - (btn_w * 2 + gap)) // 2  
    
    btn1_x1, btn1_x2 = start_x, start_x + btn_w
    btn2_x1, btn2_x2 = btn1_x2 + gap, btn1_x2 + gap + btn_w
    
    btn_y1, btn_y2 = 260, 260 + btn_h

    print("\n=== [음악 선택 메뉴 진입] ===")

    while selected_music is None:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            return "classic.mp3"  # 오류 시 기본값 반환
        
        # 1. 안내 타이틀 배경 및 텍스트박스 그리기
        title_box_x1, title_box_y1 = 150, 100
        title_box_x2, title_box_y2 = 490, 190
        cv.rectangle(frame, (title_box_x1, title_box_y1), (title_box_x2, title_box_y2), (20, 20, 20), -1) 
        cv.rectangle(frame, (title_box_x1, title_box_y1), (title_box_x2, title_box_y2), (255, 215, 0), 3) 
        
        title_txt = "SELECT MUSIC"
        txt_w, txt_h = cv.getTextSize(title_txt, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv.putText(frame, title_txt, (title_box_x1 + (340 - txt_w)//2, title_box_y1 + (90 + txt_h)//2), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. 핸드 제스처 및 마우스(손가락 좌표) 오버랩 감지
        detected_hands = analyze_hand_gesture_mp(frame)
        active_btn_idx = 0 
        current_hand_type_found = None
        h_cx, h_cy = None, None

        for (cx, cy, current_hand_type) in detected_hands:
            if cx is not None:
                h_cx, h_cy = cx, cy
                current_hand_type_found = current_hand_type
                # 손가락 좌표가 버튼 Y축 영역 내에 있을 때
                if btn_y1 <= cy <= btn_y2:
                    if btn1_x1 <= cx <= btn1_x2: 
                        active_btn_idx = 1
                    elif btn2_x1 <= cx <= btn2_x2: 
                        active_btn_idx = 2

        # 3. 버튼 상태별 색상 매핑 (선택 영역 진입 시 초록색 하이라이트)
        colors = [(255, 0, 0), (255, 0, 0)]  # 기본 파란색 (OpenCV는 BGR 이므로 파란색)
        if active_btn_idx > 0:
            colors[active_btn_idx - 1] = (0, 255, 0)  # 타겟 버튼 초록색 변경

        # [버튼 1: POP] 크기 및 텍스트 매핑
        cv.rectangle(frame, (btn1_x1, btn_y1), (btn1_x2, btn_y2), colors[0], 2 if active_btn_idx != 1 else 4)
        pop_w = cv.getTextSize("pop", cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv.putText(frame, "pop", (btn1_x1 + (btn_w - pop_w)//2, btn_y1 + 38), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # [버튼 2: CLASSIC] 크기 및 텍스트 매핑
        cv.rectangle(frame, (btn2_x1, btn_y1), (btn2_x2, btn_y2), colors[1], 2 if active_btn_idx != 2 else 4)
        classic_w = cv.getTextSize("classic", cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv.putText(frame, "classic", (btn2_x1 + (btn_w - classic_w)//2, btn_y1 + 38), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4. 제스처 변경 트리거를 통한 음악 확정 선택 이벤트 처리
        if current_hand_type_found and h_cx is not None:
            dot_color = (0, 255, 0) if current_hand_type_found == "HAND" else (0, 0, 255)
            cv.circle(frame, (h_cx, h_cy), 8, dot_color, -1)
            cv.putText(frame, current_hand_type_found, (h_cx - 20, h_cy - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 2)

            # 주먹 쥐기 혹은 손펴기 등 제스처가 전환되었을 때 실행
            if prev_hand_type in ["FIST", "HAND"] and prev_hand_type != current_hand_type_found:
                if active_btn_idx == 1:
                    selected_music = "pop.mp3"
                elif active_btn_idx == 2:
                    selected_music = "classic.mp3"
            
            prev_hand_type = current_hand_type_found
        else:
            prev_hand_type = None

        cv.imshow(win_name, frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            return "classic.mp3"  # 강제 종료 시 기본 클래식 반환

    print(f"-> 선택된 음악: {selected_music}")
    return selected_music


# -------------------------------------------------------------------------
# 설정 및 초기화
# -------------------------------------------------------------------------

pygame.mixer.init()
cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
M, M_inv, M_final_overlay = game_settings(cap)

play_game(cap, M, M_inv, M_final_overlay)

cap.release()
cv.destroyAllWindows()