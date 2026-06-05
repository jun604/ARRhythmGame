import cv2 as cv
import numpy as np
import random
import datetime
import librosa
import time
import pygame
import mediapipe as mp
#import mediapipe.python.solutions.hands as mp_hands

# 전역 변수 설정
RESOLUTION = (640, 480)
GAME_BOARD_SIZE = (640, 480) # 가상의 플랫한 게임 보드 크기

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
# max_num_hands=2 로 설정하여 최대 2개의 손을 동시에 인식하게 만듭니다.
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
    사용자가 스페이스바를 누르거나, 화면에 감지된 손들 중 하나가 
    가이드 박스 안에서 제스처를 바꿀 때 조건이 충족되어 사진을 확정합니다.
    """
    btn_y1, btn_y2 = 180, 300

    # --- 손 상태(피부색) 및 제스처 분석 ---
    detected_hands = analyze_hand_gesture_mp(frame)
    
    gesture_changed = False
    is_hand_in_zone = False
    best_current_hand_type = None

    # 감지된 모든 손을 하나씩 순회하며 가이드 박스 체크 및 시각화
    for (cx, cy, current_hand_type) in detected_hands:
        if cx is not None and current_hand_type in ["FIST", "HAND"]:
            # 현재 검사 중인 손이 START ZONE 박스 안에 들어와 있는가?
            if btn_x1 <= cx <= btn_x2 and btn_y1 <= cy <= btn_y2:
                is_hand_in_zone = True
                best_current_hand_type = current_hand_type # 박스 안의 손 상태를 기준 제스처로 기록
                
                # 가이드 박스 안의 손 시각화
                color = (0, 255, 0) if current_hand_type == "HAND" else (0, 0, 255)
                cv.circle(frame, (cx, cy), 10, color, -1)
                cv.putText(frame, current_hand_type, (cx - 30, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 박스 안에서 이전 제스처와 달라졌다면 트리거 발동!
                if prev_hand_type in ["FIST", "HAND"] and prev_hand_type != current_hand_type:
                    gesture_changed = True
                    print(f"✊✋ 버튼 영역 내 제스처 변경 감지! ({prev_hand_type} -> {current_hand_type})")
            else:
                # 박스 밖에 있는 손도 화면에 위치 정보는 그려줍니다. (피드백용)
                cv.circle(frame, (cx, cy), 6, (255, 255, 0), -1)

    # 💡 가이드 박스 안에 있는 손을 기준으로 다음 프레임의 prev_hand_type을 갱신
    if is_hand_in_zone:
        prev_hand_type = best_current_hand_type
    else:
        prev_hand_type = None

    # --- 화면 안내 메시지 및 START ZONE 시각화 ---
    cv.putText(frame, "Change Gesture in BOX or Press [Spacebar]", (30, 50), 
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    box_color = (0, 255, 0) if is_hand_in_zone else (255, 0, 0)
    box_thickness = 4 if is_hand_in_zone else 2
    cv.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), box_color, box_thickness)
    cv.putText(frame, "START ZONE", (btn_x1 - 10, btn_y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    # --- 입력 트리거 판정 ---
    key = cv.waitKey(1) & 0xFF
    if key == ord(' ') or gesture_changed:
        if win_name != "AR Camera (Floor Scan Mode)":
            cv.destroyWindow(win_name) 
        prev_hand_type = None
        return frame, prev_hand_type, gesture_changed # 확실히 결정되었으므로 True 반환

    elif key == ord('q'):
        cv.destroyWindow(win_name)
        exit()
    
    else:
        return frame, prev_hand_type, gesture_changed # 아직 결정되지 않았으므로 False 반환

def find_flat(cap, ref_img, game_board_size=GAME_BOARD_SIZE):
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
    new_frame = None
    prev_hand_type = None
    gesture_changed = False

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
                    cv.putText(frame, "MATCH FOUND! select frame", (30, 50), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                               #select_picture을 영상이 아닌 frame단위로 바꾸면 가능?

                    src_points = dynamic_src_points.reshape(4, 2)
                    dst_points = np.float32([
                        [0, 0], 
                        [game_board_size[0], 0], 
                        [game_board_size[0], game_board_size[1]], 
                        [0, game_board_size[1]]
                    ])
                    M = cv.getPerspectiveTransform(src_points, dst_points)
                    if M is not None and gesture_changed:
                        print("-> 바닥 좌표 동적 추출 성공! 게임을 시작합니다.")
                        cv.destroyWindow(win_name)
                        return M
            else:
                cv.putText(frame, "Scanning floor marker...", (30, 50), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.imshow(win_name, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return None

def analyze_hand_gesture_mp(frame):
    """
    구글 MediaPipe 딥러닝 모델을 사용하여 최대 2개의 손을 인식하고 
    [(cx, cy, hand_type), ...] 리스트를 반환합니다.
    """
    h, w, _ = frame.shape
    # MediaPipe는 RGB 이미지를 사용하므로 변환
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    hands_info = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 💡 0번(손목)과 9번(가운데 손가락 시작점) 좌표를 이용해 손의 중심점(cx, cy) 계산
            cx = int(hand_landmarks.landmark[9].x * w)
            cy = int(hand_landmarks.landmark[9].y * h)
            
            # --- 💡 초간단 주먹/보(FIST/HAND) 판정 알고리즘 ---
            # 엄지를 제외한 네 손가락(끝마디: 8, 12, 16, 20)이 
            # 각각 안쪽 마디(6, 10, 14, 18)보다 Y값이 아래에 있으면 접힌 것(주먹)으로 판정합니다.
            folded_fingers = 0
            tip_ids = [8, 12, 16, 20]
            pip_ids = [6, 10, 14, 18]
            
            for tip, pip in zip(tip_ids, pip_ids):
                if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
                    folded_fingers += 1
            
            # 접힌 손가락이 3개 이상이면 주먹(FIST), 아니면 보(HAND)
            hand_type = "FIST" if folded_fingers >= 3 else "HAND"
            
            hands_info.append((cx, cy, hand_type))
            
    return hands_info

def make_notes(AUDIO_FILE):
    print(f"🎵 [librosa 오디오 분석 중] '{AUDIO_FILE}' 채보를 자동 생성하고 있습니다...")
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
                
        print(f"✅ 채보 생성 완료! 총 {len(generated_notes)}개의 비트 노트를 배치했습니다.")
        return generated_notes
    except Exception as e:
        print(f"❌ 오디오 분석 중 오류 발생: {e}")
        print("-> 기본 데모 채보(BPM 120 기준)로 대체합니다.")
        return [[i * 0.5, random.randint(0, 3), False] for i in range(2, 100)]

# -------------------------------------------------------------------------
# 설정 및 초기화
# -------------------------------------------------------------------------
lane_x = [80, 240, 400, 560] 
score = 0
AUDIO_FILE = "classic.mp3" 

# pygame 오디오 믹서 초기화 및 음악 로드
pygame.mixer.init()
pygame.mixer.music.load(AUDIO_FILE)

game_notes = make_notes(AUDIO_FILE)
active_notes = [] 

NOTE_SPEED = 400   
JUDGE_LINE_Y = 420 
lead_time = JUDGE_LINE_Y / NOTE_SPEED 

cap = cv.VideoCapture(1) # 사용자 환경에 맞게 0 또는 1 설정
cap.set(cv.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

ret, old_frame = cap.read()
if not ret:
    print("카메라를 열 수 없습니다.")
    exit()

prev_hand_type = None
gesture_changed = False
win_name = "AR Camera (Initialization)"
ret, frame = cap.read()
if not ret:
    print("카메라를 열 수 없습니다.")
    exit()

while not gesture_changed:
    frame, prev_hand_type, gesture_changed = select_picture(frame, prev_hand_type, 0, 140, win_name=win_name)
    cv.imshow(win_name, frame)
    ret, frame = cap.read()
    if not ret:
        print("카메라를 열 수 없습니다.")
        exit()
cv.destroyWindow(win_name)

M = find_flat(cap, frame, GAME_BOARD_SIZE)
if M is None:
    print("바닥 스캔 실패")
    exit()

# 원래 화면으로 되돌리기 위한 역행렬(Inverse Matrix) 계산
M_inv = np.linalg.inv(M)

print("\n=== [2단계: 게임 시작] ===")

# 제스처 실시간 변화 감지를 위한 상태 저장 변수
prev_hand_type = None

pygame.mixer.music.play(0)
game_start_time = time.time() 

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    elapsed_time = pygame.mixer.music.get_pos() / 1000.0
    
    # 1. 검은색 도화지(가상 게임판)를 만들어 평면 기준 노트를 그리기
    # 왜곡 변환(Warp) 시 검은색 영역만 카메라 화면에 자연스럽게 합성하기 위함입니다.
    virtual_board = np.zeros((GAME_BOARD_SIZE[1], GAME_BOARD_SIZE[0], 3), dtype=np.uint8)
    
    # --- 노트 스폰 메커니즘 ---
    for note in game_notes:
        target_time, lane, is_spawned = note
        if not is_spawned and elapsed_time >= (target_time - lead_time):
            active_notes.append([lane_x[lane], 0, lane, target_time])
            note[2] = True 
        
    # 가상 보드에 판정선 그리기
    cv.line(virtual_board, (0, JUDGE_LINE_Y), (640, JUDGE_LINE_Y), (255, 0, 0), 5)
    
    # 가상 보드에 노트 그리기
    for note in active_notes[:]:
        time_to_target = note[3] - elapsed_time
        note[1] = int(JUDGE_LINE_Y - (time_to_target * NOTE_SPEED))
        
        if note[1] > 460:
            active_notes.remove(note)
            print("MISS!")
            continue
            
        # 가상 평면 보드에 빨간색 노트 사각형 렌더링
        cv.rectangle(virtual_board, (note[0]-80, note[1]-20), (note[0]+80, note[1]+20), (0, 0, 255), -1)

    # 2. 역변환 행렬(M_inv)을 사용해 평면 게임판을 원래 카메라 시점으로 찌그러뜨리기(Warp)
    warped_game_overlay = cv.warpPerspective(virtual_board, M_inv, RESOLUTION)

    # 3. 실시간 카메라 화면(frame)에 왜곡된 게임판 레이어를 투명하게 합성 (비트 연산 활용)
    # 노트가 그려진 부분만 뚫어서 카메라 영상에 덮어씌웁니다.
    overlay_gray = cv.cvtColor(warped_game_overlay, cv.COLOR_BGR2GRAY)
    _, mask_inv = cv.threshold(overlay_gray, 1, 255, cv.THRESH_BINARY_INV)
    
    # 가상 노트 레이어가 들어갈 자리를 카메라 프레임에서 도려냄
    background = cv.bitwise_and(frame, frame, mask=mask_inv)
    # 도려낸 자리에 왜곡된 가상 레이어를 병합하여 최종 AR 프레임 생성
    ar_frame = cv.add(background, warped_game_overlay)

    # 4. 손 상태(피부색) 및 제스처 분석    
    detected_hands = analyze_hand_gesture_mp(frame)
    
    # 여러 손의 제스처 변화를 통합 추적하기 위해 기존 단일 변수 대신 세트로 관리하거나, 
    # 혹은 단순 터치용도라면 루프 내에서 개별 판단합니다.
    # (여기서는 단순 터치/변화 트리거 판정을 위해 개별 처리 예시를 듭니다)
    
    current_hand_types = [] # 이번 프레임에 발견된 제스처 목록
    
    for (cx, cy, current_hand_type) in detected_hands:
        current_hand_types.append(current_hand_type)
        
        # Y축 판정선 스냅(보정) 알고리즘 적용
        threshold_y = 120 
        if abs(cy - JUDGE_LINE_Y) <= threshold_y:
            cy = JUDGE_LINE_Y

        # 각 손 위치 시각화
        color = (0, 255, 0) if current_hand_type == "HAND" else (0, 0, 255)
        cv.circle(ar_frame, (cx, cy), 10, color, -1) 
        cv.putText(ar_frame, current_hand_type, (cx - 30, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 카메라 좌표 -> 가상 평면 좌표 매핑
        hand_point = np.array([[[cx, cy]]], dtype=np.float32)
        transformed_hand = cv.perspectiveTransform(hand_point, M)
        hx, hy = transformed_hand[0][0][0], transformed_hand[0][0][1]
        
        # 💡 각 손의 제스처 상태 변화 감지 (단일 변수 대신 단순화하거나 이전 frame 핸드 타입과 매칭 필요)
        # 양손의 경우 완벽한 추적을 위해선 좌/우 구분이 필요하나, 
        # 리듬게임 특성상 "박자 순간에 제스처가 변했는가?"가 중요하므로 
        # 이전 프레임에 없던 제스처가 생겼거나 변했을 때 트리거를 켜줍니다.
        gesture_changed = False
        if prev_hand_type is not None and prev_hand_type != current_hand_type:
            gesture_changed = True
            
        # 5. 제스처 변경 트리거 시 해당 손의 위치(hx)로 노트 판정
        if gesture_changed and 0 <= hx < GAME_BOARD_SIZE[0]:
            for note in active_notes[:]:
                if (abs(hx - note[0]) < 60) and (abs(elapsed_time - note[3]) < 0.15):
                    score += 100
                    print(f"🎯 PERFECT HIT! SCORE: {score}")
                    active_notes.remove(note)

    # 다음 프레임을 위해 저장 (가장 대표적인 첫 번째 손의 제스처 위주로 저장하거나 리스트 백업)
    if current_hand_types:
        prev_hand_type = current_hand_types[0]
    else:
        prev_hand_type = None

    # 스코어 표시 및 최종 결과물 출력
    cv.putText(ar_frame, f"SCORE: {score}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("AR Rhythm Game Play Board (Camera View)", ar_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()