import cv2 as cv
import numpy as np
import random
import datetime

# 전역 변수 설정 (세로 크기를 700으로 통일)
RESOLUTION = (640, 480)
GAME_BOARD_SIZE = (400, 700) 

# 게임 노트를 관리할 리스트 [x_pos, y_pos, lane]
notes = []
score = 0
lane_x = [50, 150, 250, 350] # 4개 라인의 X 좌표 (게임 보드 기준)

# Optical Flow 파라미터
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv.VideoCapture(1) # 일반적인 웹캠 기본값 0으로 변경 (필요시 1로 수정)
cap.set(cv.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

ret, old_frame = cap.read()
if not ret:
    print("카메라를 열 수 없습니다.")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

is_scanned = False
M = None 

print("=== [1단계: 바닥 스캔] ===")
print("카메라를 바닥을 향해 좌우로 천천히 움직이세요.")
print("바닥에 초록색 점들이 안정적으로 생기면 [Spacebar]를 눌러 바닥을 확정하세요.")
#out_recorder = cv.VideoWriter("Video_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi", cv.VideoWriter_fourcc(*'XVID'), 30.0, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
#play_recorder = cv.VideoWriter("Play_Video_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi", cv.VideoWriter_fourcc(*'XVID'), 30.0, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #if is_record:
        record_video(frame, play_recorder)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # --- 1단계: 바닥 스캔 모드 ---
    if not is_scanned:
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                cv.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
                
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            
            if len(p0) < 20:
                p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        else:
            p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            
        cv.putText(frame, "Scan Floor & Press [Space]", (30, 50), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.imshow("AR Camera (Original)", frame)
        
        key = cv.waitKey(30)
        if key == ord(' '): 
            if len(p0) >= 4:
                # 사용자가 지정한 사다리꼴 구역을 400x700 직사각형으로 맵핑
                src_points = np.float32([[220, 100], [420, 100], [20, 500], [620, 500]])
                dst_points = np.float32([[0, 300], [400, 300], [0, 600], [400, 600]]) # Y축 시작을 0으로 수정
                M = cv.getPerspectiveTransform(src_points, dst_points)
                is_scanned = True
                print("\n=== [2단계: 게임 시작] ===")
            else:
                print("특징점이 부족합니다. 바닥을 더 비춰주세요.")
        elif key == ord('q'):
            break

    # --- 2단계: 게임 플레이 모드 ---
    else:
        # 1. 원본 화면을 호모그래피 Matrix로 펼치기
        warped_floor = cv.warpPerspective(frame, M, GAME_BOARD_SIZE)
        
        # 2. 주기적으로 랜덤하게 노트 생성
        if random.random() < 0.15 and len(notes) < 5:
            lane = random.randint(0, 3)
            notes.append([lane_x[lane], 0, lane]) 
            
        # 3. 게임 보드에 판정선 그리기 (Y=550 지점)
        JUDGE_LINE_Y = 550
        cv.line(warped_floor, (0, JUDGE_LINE_Y), (400, JUDGE_LINE_Y), (255, 0, 0), 3)
        
        # 4. 노트 업데이트 및 그리기
        for note in notes[:]:
            note[1] += 10 # 낙하 속도 약간 상향
            
            # 노트 렌더링
            cv.rectangle(warped_floor, (note[0]-30, note[1]-15), (note[0]+30, note[1]+15), (0, 0, 255), -1)
            
            # 판정선을 완전히 지나치면(예: Y=600) MISS 처리
            if note[1] > 600:
                notes.remove(note)
                print("MISS!")

        # 5. 손 검출 및 좌표 변환
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv.inRange(hsv, lower_skin, upper_skin)
        
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv.contourArea)
            if cv.contourArea(max_contour) > 2000: 
                extBottom = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
                cv.circle(frame, extBottom, 10, (0, 0, 255), -1) 
                
                # 원본 좌표 -> 게임 보드 좌표계 변환
                hand_point = np.array([[[extBottom[0], extBottom[1]]]], dtype=np.float32)
                transformed_hand = cv.perspectiveTransform(hand_point, M)
                hx, hy = transformed_hand[0][0][0], transformed_hand[0][0][1]
                
                # 게임판 내에 손이 들어왔을 때
                if 0 <= hx < GAME_BOARD_SIZE[0] and 0 <= hy < GAME_BOARD_SIZE[1]:
                    cv.circle(warped_floor, (int(hx), int(hy)), 15, (0, 255, 255), -1)
                    
                    # 충돌 판정 완화: 손이 판정선 근처에 있고, 노트가 판정선 오차 범위(±30px)에 있을 때
                    for note in notes[:]:
                        if (JUDGE_LINE_Y - 40 <= hy <= JUDGE_LINE_Y + 40) and \
                           (abs(hx - note[0]) < 50) and \
                           (JUDGE_LINE_Y - 30 <= note[1] <= JUDGE_LINE_Y + 30):
                            score += 100
                            print(f"PERFECT! SCORE: {score}")
                            notes.remove(note)

        # 화면 출력
        cv.putText(warped_floor, f"SCORE: {score}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("AR Camera (Original)", frame)
        cv.imshow("AR Rhythm Game Play Board", warped_floor)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()