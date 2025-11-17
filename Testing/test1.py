import cv2
from humanoid_library import (
    load_image,
    preprocess_image,
    PoseExtractor,
    select_main_skeleton_multiple,
)

# Path to your test image
image_path = "img1.jpg"
save_path = "img1_skeleton.jpg"

# 1️⃣ Load image
image = load_image(image_path)

# 2️⃣ Preprocess (resize if needed)
image_resized = preprocess_image(image, target_size=(512, 512))

# 3️⃣ Initialize Mediapipe Pose extractor
pose_extractor = PoseExtractor()

# 4️⃣ Extract skeleton
skeleton = pose_extractor.extract_keypoints(image_resized)

if skeleton is None or len(skeleton) == 0:
    print("No skeleton detected!")
else:
    # 5️⃣ Draw skeleton and save
    pose_extractor.draw_skeleton(image_resized, skeleton, save_path)
    print(f"Skeleton drawn and saved to {save_path}")

    # Optional: show image in window
    cv2.imshow("Skeleton Overlay", cv2.cvtColor((image_resized*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
