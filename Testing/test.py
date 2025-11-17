from humanoid_library import (
    load_image,
    preprocess_image,
    PoseExtractor,
    compute_joint_angles_improved,
    select_main_skeleton_multiple,
)

image1_path = "img3.jpg"
save1_path = "out3.jpg"

# load and process the image
image1 = load_image(image1_path)
image1r = preprocess_image(image1,(640,480))

# instantiate the pose extractor class
extractor = PoseExtractor()

main_skel1 = select_main_skeleton_multiple(extractor,image1r,save1_path)
print("main_skel1: ",main_skel1)
if main_skel1 is None:
    print("No person in image1")
else:
    angles1 = compute_joint_angles_improved(main_skel1)
    print("The joint angles are: ",angles1)
