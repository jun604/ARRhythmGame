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
VIRTUAL_BOARD_SIZE = (480, 600) # 노트가 흘러내려오는 원본 긴 보드 크기

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

def record_video(frame, recorder):
    if recorder is not None:
        recorder.write(frame)

def select_picture(frame, prev_hand_type=None, btn_x1=520, btn_x2=640, win_name="Select Picture"):
    """
    사용자가 스페이스바를 누르거나 손의 제스처를 바꿀 때 조건이 충족되어 사진을 확정합니다.
    """
    btn_y1, btn_y2 = 180, 300
    
    # -----------------------------------------------------------------
    # 수정사항 1: 메인 초기화 및 바닥 스캔 단계 모두에서 우측 480x480 박스 가이드라인 상시 표시
    # -----------------------------------------------------------------
    guide_pts = np.array([[160, 0], [640, 0], [640, 480], [160, 480]], dtype=np.int32)
    cv.polylines(frame, [guide_pts], True, (255, 255, 0), 2)
    cv.putText(frame, "Align Marker inside this 480x480 BOX", (170, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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
                cv.putText(frame, current_hand_type, (cx - 30, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
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
        return frame, prev_hand_type, True # 제스처 변경이나 스페이스바 입력 시 True 반환하도록 보장

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

    kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
    h_ref, w_ref = ref_img.shape[:2]
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

# -------------------------------------------------------------------------
# 설정 및 초기화
# -------------------------------------------------------------------------
lane_x = [60, 180, 300, 420] 
score = 0
AUDIO_FILE = "classic.mp3" 

pygame.mixer.init()
pygame.mixer.music.load(AUDIO_FILE)

game_notes = make_notes(AUDIO_FILE)
active_notes = [] 

NOTE_SPEED = 400   

JUDGE_LINE_Y = VIRTUAL_BOARD_SIZE[1] - 80  # 600 - 80 = 520
lead_time = JUDGE_LINE_Y / NOTE_SPEED 

# -------------------------------------------------------------------------
# 수정사항 2-1: 개별 노트 위치에 텍스트를 띄우기 위한 이펙트 큐(List) 구조 도입
# -------------------------------------------------------------------------
# 효과 여러 개가 개별 위치에 머물다 사라지도록 관리합니다.
# 각 원소 형태: {"text": "hit", "color": (0,0,255), "expire_time": 시간, "v_x": 가상보드X, "v_y": 가상보드Y}
active_effects = []

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

ret, frame = cap.read()
if not ret:
    print("카메라를 열 수 없습니다.")
    exit()

prev_hand_type = None
gesture_changed = False
win_name = "AR Camera (Initialization)"

# 첫 메인 진입 루프구간 - 수정사항 1 적용되어 우측 안내선이 출력됨
while not gesture_changed:
    frame, prev_hand_type, gesture_changed = select_picture(frame, prev_hand_type, 0, 140, win_name=win_name)
    cv.imshow(win_name, frame)
    ret, frame = cap.read()
    if not ret:
        exit()
cv.destroyWindow(win_name)

# 480x480 기반으로 바닥 호모그래피 행렬 M 계산
M = find_flat(cap, frame, SCAN_BOARD_SIZE)
if M is None:
    print("바닥 스캔 실패")
    exit()

# 투영 변환(Warp)을 위한 커스텀 왜곡 매트릭스 계산
src_project_pts = np.float32([
    [0, VIRTUAL_BOARD_SIZE[1] - SCAN_BOARD_SIZE[1]],                  # (0, 120)
    [SCAN_BOARD_SIZE[0], VIRTUAL_BOARD_SIZE[1] - SCAN_BOARD_SIZE[1]], # (480, 120)
    [SCAN_BOARD_SIZE[0], VIRTUAL_BOARD_SIZE[1]],                      # (480, 600)
    [0, VIRTUAL_BOARD_SIZE[1]]                                        # (0, 600)
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

print("\n=== [2단계: 게임 시작] ===")
pygame.mixer.music.play(0)
game_start_time = time.time() 

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    elapsed_time = pygame.mixer.music.get_pos() / 1000.0
    current_time = time.time()
    
    # 세로로 긴 (480, 600) 가상 게임판 도화지 생성
    virtual_board = np.zeros((VIRTUAL_BOARD_SIZE[1], VIRTUAL_BOARD_SIZE[0], 3), dtype=np.uint8)
    
    # 노트 스폰 메커니즘
    for note in game_notes:
        target_time, lane, is_spawned = note
        if not is_spawned and elapsed_time >= (target_time - lead_time):
            active_notes.append([lane_x[lane], 0, lane, target_time])
            note[2] = True 
        
    # 가상 보드 판정선 그리기 (Y = 520)
    cv.line(virtual_board, (0, JUDGE_LINE_Y), (VIRTUAL_BOARD_SIZE[0], JUDGE_LINE_Y), (255, 0, 0), 5)
    
    # 가상 보드 노트 렌더링 및 탈락 처리
    for note in active_notes[:]:
        time_to_target = note[3] - elapsed_time
        note[1] = int(JUDGE_LINE_Y - (time_to_target * NOTE_SPEED))
        
        # 미스 처리 (바닥 면을 벗어남)
        if note[1] > VIRTUAL_BOARD_SIZE[1] - 10:
            # -----------------------------------------------------------------
            # 수정사항 2-2: 미스한 노트가 있던 X좌표 바닥 부근에 회색 "miss" 추가
            # -----------------------------------------------------------------
            active_effects.append({
                "text": "miss",
                "color": (128, 128, 128),
                "expire_time": current_time + 0.5,
                "v_x": note[0] - 35,
                "v_y": VIRTUAL_BOARD_SIZE[1] - 30 # 가상 보드 아래쪽 끝 지점
            })
            active_notes.remove(note)
            continue
            
        cv.rectangle(virtual_board, (note[0]-50, note[1]-15), (note[0]+50, note[1]+15), (0, 0, 255), -1)

    # -----------------------------------------------------------------
    # 수정사항 2-3: 활성화된 이펙트 글자들을 가상 보드(virtual_board) 위에 직접 드로잉
    # 이렇게 하면 호모그래피 변환을 함께 타기 때문에 실제 3D 게임판 상의 해당 노트 위치에 왜곡되어 정확히 올라갑니다!
    # -----------------------------------------------------------------
    for fx in active_effects[:]:
        if current_time > fx["expire_time"]:
            active_effects.remove(fx)
        else:
            cv.putText(virtual_board, fx["text"].upper(), (fx["v_x"], fx["v_y"]), 
                       cv.FONT_HERSHEY_DUPLEX, 0.8, fx["color"], 2)

    # 투영 변환 및 카메라 영상 합성
    warped_game_overlay = cv.warpPerspective(virtual_board, M_final_overlay, RESOLUTION)
    overlay_gray = cv.cvtColor(warped_game_overlay, cv.COLOR_BGR2GRAY)
    _, mask_inv = cv.threshold(overlay_gray, 1, 255, cv.THRESH_BINARY_INV)
    
    background = cv.bitwise_and(frame, frame, mask=mask_inv)
    ar_frame = cv.add(background, warped_game_overlay)

    # 손 제스처 분석 및 인터랙션 처리    
    detected_hands = analyze_hand_gesture_mp(frame)
    current_hand_types = []
    
    for (cx, cy, current_hand_type) in detected_hands:
        current_hand_types.append(current_hand_type)
        
        hand_point = np.array([[[cx, cy]]], dtype=np.float32)
        transformed_hand = cv.perspectiveTransform(hand_point, M)
        hx, hy = transformed_hand[0][0][0], transformed_hand[0][0][1]
        
        scan_judge_y = SCAN_BOARD_SIZE[1] - 80 
        
        if abs(hy - scan_judge_y) <= 80:
            corrected_hand = cv.perspectiveTransform(np.array([[[hx, scan_judge_y]]], dtype=np.float32), M_inv)
            cx_draw, cy_draw = int(corrected_hand[0][0][0]), int(corrected_hand[0][0][1])
        else:
            cx_draw, cy_draw = cx, cy

        color = (0, 255, 0) if current_hand_type == "HAND" else (0, 0, 255)
        cv.circle(ar_frame, (cx_draw, cy_draw), 10, color, -1) 
        cv.putText(ar_frame, current_hand_type, (cx_draw - 30, cy_draw - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        gesture_changed = False
        if prev_hand_type is not None and prev_hand_type != current_hand_type:
            gesture_changed = True
            
        # 판정 처리
        if gesture_changed and 0 <= hx < SCAN_BOARD_SIZE[0]:
            for note in active_notes[:]:
                if (abs(hx - note[0]) < 50) and (abs(elapsed_time - note[3]) < 0.15):
                    score += 100
                    
                    # -------------------------------------------------------------
                    # 수정사항 2-4: 맞춘 노트의 바로 위쪽(판정선 살짝 상단)에 빨간색 "hit" 추가
                    # -------------------------------------------------------------
                    active_effects.append({
                        "text": "hit",
                        "color": (0, 0, 255), # 빨간색
                        "expire_time": current_time + 0.5,
                        "v_x": note[0] - 25,
                        "v_y": JUDGE_LINE_Y - 20 # 판정선보다 20 픽셀 위
                    })
                    active_notes.remove(note)

    if current_hand_types:
        prev_hand_type = current_hand_types[0]
    else:
        prev_hand_type = None

    # 스코어 UI 출력
    cv.putText(ar_frame, f"SCORE: {score}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("AR Rhythm Game Play Board (Camera View)", ar_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()