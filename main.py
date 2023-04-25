import numpy as np
import cv2, math
from background import *
from vehicle_detector import *
from matplotlib import path

backgroundFrame = Video.bg_frame("traffic.mp4")

size = backgroundFrame.shape[:2][::-1]

x, _ = frame_width, frame_height = size

half = int(frame_height / 2)
roi = backgroundFrame[half:, :]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 200, 300)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, maxLineGap=100)

l1 = l2 = None

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if (x1 + x2) / 2 < x:
            l1 = line[0]
            x = (x1 + x2) / 2

    yt = frame_height
    xt, xb, yb = 0, 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if 100 + (l1[0] + l1[2]) / 2 < (x1 + x2) / 2 < 350 + (l1[0] + l1[2]) / 2:
            if yt > y2:
                yt = y2
                xt = x2

            if yb < y1:
                yb = y1
                xb = x1

    l2 = (xb, yb, xt, yt)
p = path.Path([(0, half + l2[1]), (l2[0], half + l2[1]), (l2[2], half + l2[3]), (l2[2], 0), (0, 0), (0, half + l2[3])])

video = cv2.VideoCapture("traffic.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
check_frames = (fps * 1)/2
print(check_frames)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
result = cv2.VideoWriter('output.mp4', fourcc, fps, size)

count = 1
center_points_prev_frame = []

tracking_objects = {}
track_id = 0
object_id = None

steady_cars = []
steady_cars_frames = []
steady_cars_pos = []

vd = VehicleDetector()
while True:
    ret, frame = video.read()
    if not ret:
        break

    center_points_cur_frame = []

    vehicle_boxes = vd.detect_vehicles(frame[0:, :max(l2[0], l2[2])])

    for box in vehicle_boxes:
        x, y, w, h = box

        cx = x + int(w / 2)
        cy = 0 + y + int(h / 2)

        if p.contains_points([(cx, cy)])[0]:
            center_points_cur_frame.append((cx, cy, w, h))

    if count <= 2:
        for pt2 in center_points_prev_frame:
            for pt in center_points_cur_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 30:
                    if distance < 3:
                        tracking_objects[track_id] = [pt, tracking_objects.get(track_id, (pt, count))[1]]
                    else:
                        tracking_objects[track_id] = [pt, count]
                    track_id += 1
                    continue
        center_points_prev_frame = center_points_cur_frame.copy()

    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2_pos in tracking_objects_copy.items():
            object_exists = False

            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2_pos[0][0] - pt[0], pt2_pos[0][1] - pt[1])

                if distance < 30:
                    if distance < 3:
                        tracking_objects[object_id] = [pt, tracking_objects.get(object_id, (pt, count))[1]]
                    else:
                        tracking_objects[object_id] = [pt, count]
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            if not object_exists:
                tracking_objects.pop(object_id)

        for pt in center_points_cur_frame:
            tracking_objects[track_id] = [pt, tracking_objects.get(track_id, (pt, count))[1]]
            track_id += 1

    for object_id, pt_pos in tracking_objects.items():
        if p.contains_points([pt_pos[0][:2]])[0]:
            pt1 = (pt_pos[0][0] - int(pt_pos[0][2]/2), 0 + pt_pos[0][1] - int(pt_pos[0][3]/2))
            pt2 = (pt_pos[0][0] + int(pt_pos[0][2]/2), 0 + pt_pos[0][1] + int(pt_pos[0][3]/2))
            if count-pt_pos[1] >= check_frames:
                if object_id not in steady_cars:
                    steady_cars.append(object_id)
                    steady_cars_frames.append([i for i in range(count - 10, count + 10)])
                    steady_cars_pos.append(pt_pos[0])

                cv2.putText(frame, str(object_id + 1), (pt_pos[0][0] - 7, pt_pos[0][1] - 0 + 7), 0, 1, (0, 0, 0), 2)
                cv2.rectangle(frame, pt1, pt2, (0, 0, 0), 3)
            else:
                cv2.putText(frame, str(object_id + 1), (pt_pos[0][0] - 7, pt_pos[0][1] - 0 + 7), 0, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 3)

    cv2.line(frame, (l2[0], half + l2[1]), (l2[2], half + l2[3]), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    # cv2.imshow("edges", edges)
    result.write(frame)
    print(count)

    key = cv2.waitKey(1)
    if key == 27:
        break

    count += 1

steady_frames(steady_cars, steady_cars_frames, steady_cars_pos, cv2.VideoCapture("Traffic_Trim.mp4"))

video.release()
cv2.destroyAllWindows()
