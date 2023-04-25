import numpy as np
import cv2, os, shutil


class Video:
    @staticmethod
    def bg_frame(video_name):
        video = cv2.VideoCapture(video_name)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

        frames = []

        for i in range(30):
            video.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * (i / 30)))
            ret, frame = video.read()
            frames.append(frame)

        backgroundFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

        return backgroundFrame


def steady_frames(cars, cars_frames, pt_pos, cap):
    if cars:
        path = os.path.abspath(os.pardir)
        for i in range(len(cars)):
            folder = "carID_" + str(cars[i] + 1)
            path2 = os.path.join(path, folder)

            if os.path.exists(path2):
                shutil.rmtree(path2)

            os.mkdir(path2)

            for frame_no in cars_frames[i]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                cv2.rectangle(frame, (pt_pos[i][0] - int(pt_pos[i][2]/2), 0 + pt_pos[i][1] - int(pt_pos[i][3]/2)), (pt_pos[i][0] + int(pt_pos[i][2]/2), 0 + pt_pos[i][1] + int(pt_pos[i][3]/2)), (0, 0, 256), 3)
                cv2.imwrite(os.path.join(path2, str(frame_no + 1) + ".jpg"), frame)
    else:
        print("No steady cars so far.")
