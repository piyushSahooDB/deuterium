import pyzed.sl as sl
import cv2
import os
from datetime import datetime

# ==========================
# INIT ZED
# ==========================

zed = sl.Camera()

init = sl.InitParameters()

init.camera_resolution = sl.RESOLUTION.HD2K
init.camera_fps = 15
init.input.set_from_serial_number()
status = zed.open(init)

if status != sl.ERROR_CODE.SUCCESS:
    print("[ERROR]", status)
    exit()

print("[INFO] Camera opened")

runtime = sl.RuntimeParameters()

image = sl.Mat()

# ==========================
# CREATE SAVE FOLDERS
# ==========================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

base_folder = f"stereo_dataset_{timestamp}"

left_folder = os.path.join(base_folder,"left")
right_folder = os.path.join(base_folder,"right")

os.makedirs(left_folder,exist_ok=True)
os.makedirs(right_folder,exist_ok=True)

print(f"[INFO] Saving to {base_folder}")

count=0

# ==========================
# LOOP
# ==========================

try:

    while True:

        if zed.grab(runtime)==sl.ERROR_CODE.SUCCESS:

            zed.retrieve_image(
                image,
                sl.VIEW.SIDE_BY_SIDE
            )

            frame=image.get_data()

            frame=cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR
            )

            h,w,_=frame.shape

            mid=w//2

            left_img=frame[:,0:mid]
            right_img=frame[:,mid:w]

            display=frame.copy()

            cv2.putText(
                display,
                "SPACE: Save | Q: Quit",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            cv2.putText(
                display,
                f"Images: {count}",
                (20,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            cv2.imshow(
                "Stereo Camera",
                display
            )

            key=cv2.waitKey(1)&0xFF

            # SPACEBAR
            if key==32:

                left_name=os.path.join(
                    left_folder,
                    f"left_{count:05d}.png"
                )

                right_name=os.path.join(
                    right_folder,
                    f"right_{count:05d}.png"
                )

                cv2.imwrite(
                    left_name,
                    left_img
                )

                cv2.imwrite(
                    right_name,
                    right_img
                )

                print(
                    f"[SAVED] Pair {count}"
                )

                count+=1

            elif key==ord('q'):
                break

except KeyboardInterrupt:

    print("\nStopping...")

# ==========================
# CLEANUP
# ==========================

cv2.destroyAllWindows()

zed.close()

print(f"[INFO] Saved {count} stereo pairs")
