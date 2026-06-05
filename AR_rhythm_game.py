import cv2 as cv
import numpy as np
import random
import datetime
import librosa
import time
import pygame

# 전역 변수 설정
RESOLUTION = (640, 480)
GAME_BOARD_SIZE = (640, 480) # 가상의 플랫한 게임 보드 크기

def record_video(frame, recorder):
    if recorder is not None:
        recorder.write(frame)

def select_picture(frame, prev_hand_type=None, btn_x1=480, btn_x2=600, win_name="Select Picture"):
    """
    사용자가 스페이스바를 누르거나 가이드 박스 안에서 제스처를 바꿀 때까지 
    카메라 영상을 보여주며 대기하고, 조건이 충족되면 그 순간의 프레임을 반환합니다.
    """
    print("\n=== [0단계: 시작 사진 결정] ===")
    print("가이드 박스 안에서 제스처를 바꾸거나 [Spacebar]를 누르면 사진이 결정됩니다. (종료: q)")
    
    #prev_hand_type = None
    btn_y1, btn_y2 = 180, 300

    #while True:
    #ret, frame = cap.read()
    """if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        return None"""

    # 원본 보존을 위해 출력용 복사본 생성
    #display_frame = frame.copy()

    # --- 손 상태(피부색) 및 제스처 분석 ---
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_skin = np.array([0, 15, 40], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    skin_mask = cv.inRange(hsv, lower_skin, upper_skin)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_OPEN, kernel)
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel)
    
    cx, cy, current_hand_type = analyze_hand_gesture(skin_mask)
    
    gesture_changed = False
    is_hand_in_zone = False

    if cx is not None:
        if btn_x1 <= cx <= btn_x2 and btn_y1 <= cy <= btn_y2:
            is_hand_in_zone = True
            color = (0, 255, 0) if current_hand_type == "HAND" else (0, 0, 255)
            cv.circle(frame, (cx, cy), 10, color, -1)
            cv.putText(frame, current_hand_type, (cx - 30, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if prev_hand_type is not None and current_hand_type is not None and prev_hand_type != current_hand_type:
                gesture_changed = True
                print(f"✊✋ 버튼 영역 내 제스처 변경 감지! ({prev_hand_type} -> {current_hand_type})")
            
            prev_hand_type = current_hand_type
        else:
            prev_hand_type = None
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
            cv.destroyWindow(win_name) # 역할이 끝난 창은 닫기
        prev_hand_type=None
        return frame, prev_hand_type, gesture_changed   # 가이드라인이 없는 순수한 원본 프레임 리턴

    elif key == ord('q'):
        cv.destroyWindow(win_name)
        return None, None, gesture_changed
    
    else:
        return frame, prev_hand_type, gesture_changed

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
                        return M
            else:
                cv.putText(frame, "Scanning floor marker...", (30, 50), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.imshow(win_name, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return None

def analyze_hand_gesture(mask, min_area=3000):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
        
    max_contour = max(contours, key=cv.contourArea)
    if cv.contourArea(max_contour) < min_area:
        return None, None, None

    M_spatial = cv.moments(max_contour)
    if M_spatial["m00"] != 0:
        cx = int(M_spatial["m10"] / M_spatial["m00"])
        cy = int(M_spatial["m01"] / M_spatial["m00"])
    else:
        extBottom = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
        cx, cy = extBottom[0], extBottom[1]

    hull_indices = cv.convexHull(max_contour, returnPoints=False)
    hand_type = "FIST"

    if len(max_contour) > 3 and len(hull_indices) > 3:
        defects = cv.convexityDefects(max_contour, hull_indices)
        if defects is not None:
            finger_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))

                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57.2958

                if angle < 90 and d > 1000:
                    finger_count += 1

            if finger_count >= 2:
                hand_type = "HAND"

    return cx, cy, hand_type

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
    frame, prev_hand_type, gesture_changed = select_picture(frame, prev_hand_type, 40, 180, win_name=win_name)
    cv.imshow(win_name, frame)
    ret, frame = cap.read()
    if not ret:
        print("카메라를 열 수 없습니다.")
        exit()

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
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv.inRange(hsv, lower_skin, upper_skin)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_OPEN, kernel)
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel)
    
    cx, cy, current_hand_type = analyze_hand_gesture(skin_mask)
    
    # 제스처 상태 변화 감지 플래그 생성
    gesture_changed = False
    if cx is not None:
        # 판정선(JUDGE_LINE_Y) 근처 허용 오차 범위 설정 (예: 60픽셀)
        threshold_y = 120 
        
        if abs(cy - JUDGE_LINE_Y) <= threshold_y:
            # 손의 Y 좌표를 판정선에 수직으로 일치하도록 강제 고정!
            cy = JUDGE_LINE_Y

        color = (0, 255, 0) if current_hand_type == "HAND" else (0, 0, 255)
        cv.circle(ar_frame, (cx, cy), 10, color, -1) 
        cv.putText(ar_frame, current_hand_type, (cx - 30, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 이전 프레임과 비교해 손 모양이 바뀌었는지 체크 (Trigger 조건)
        if prev_hand_type is not None and prev_hand_type != current_hand_type:
            gesture_changed = True
            print(f"✊✋ 제스처 변경 감지! ({prev_hand_type} -> {current_hand_type})")
            
        # 손의 카메라 좌표 -> 가상 평면 좌표계로 매핑 (어떤 레인을 건드리는지 판정하기 위함)
        hand_point = np.array([[[cx, cy]]], dtype=np.float32)
        transformed_hand = cv.perspectiveTransform(hand_point, M)
        hx, hy = transformed_hand[0][0][0], transformed_hand[0][0][1]
        
        # 5. [핵심 수정] 제스처가 변하는 찰나(Trigger)에 판정 영역 계산
        if gesture_changed and 0 <= hx < GAME_BOARD_SIZE[0]:
            for note in active_notes[:]:
                # 조건 A: 가상 평면 좌표 기준으로 내 손의 X축 위치가 레인 안에 부합하고
                # 조건 B: 실제 음악 연주 시간과 노트의 목표 판정 시간 오차가 ±0.15초 이내일 때 HIT!
                if (abs(hx - note[0]) < 60) and (abs(elapsed_time - note[3]) < 0.15):
                    score += 100
                    time_error = elapsed_time - note[3]
                    print(f"🎯 PERFECT HIT! SCORE: {score} | 오차: {time_error:.3f}초")
                    active_notes.remove(note)

        # 다음 프레임을 위해 현재 상태 저장
        prev_hand_type = current_hand_type
    else:
        prev_hand_type = None # 손이 사라지면 이전 상태 초기화

    # 스코어 표시 및 최종 결과물 출력
    cv.putText(ar_frame, f"SCORE: {score}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("AR Rhythm Game Play Board (Camera View)", ar_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()