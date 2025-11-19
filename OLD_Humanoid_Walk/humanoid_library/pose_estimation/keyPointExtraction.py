import cv2
import numpy as np
import mediapipe as mp

class PoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose
        self.detector = self.pose.Pose(static_image_mode=False,
                                       model_complexity=2,
                                       enable_segmentation=False,
                                       min_detection_confidence=0.4)
        
    def extract_keypoints(self,image):
        
        results = self.detector.process((image * 255).astype(np.uint8))
        print("Mediapipe pose landmarks: ", results.pose_landmarks)
        if not results.pose_landmarks:
            return []
        
        landmarks = results.pose_landmarks.landmark
        keypoints_33 = np.array([[lm.x,lm.y, lm.z] for lm in landmarks])

        #chosen 25 points similar to the ones in body_25 model approx using the mediapipe
        #the design choise to use mediapipe is body_25 has lot of dependencies which makes it tough to install and use for the end user
        #also mediapipe is a pip installable module so user can get it through pip install -e

         # average helper
        def avg(i, j):
            p1, p2 = keypoints_33[i], keypoints_33[j]
            return (p1 + p2) / 2.0
        
        #body_25 points
        body25 = [
            keypoints_33[0],            # 0 Nose
            avg(11, 12),                # 1 Neck
            keypoints_33[12],           # 2 RShoulder
            keypoints_33[14],           # 3 RElbow
            keypoints_33[16],           # 4 RWrist
            keypoints_33[11],           # 5 LShoulder
            keypoints_33[13],           # 6 LElbow
            keypoints_33[15],           # 7 LWrist
            avg(23, 24),                # 8 MidHip
            keypoints_33[24],           # 9 RHip
            keypoints_33[26],           # 10 RKnee
            keypoints_33[28],           # 11 RAnkle
            keypoints_33[23],           # 12 LHip
            keypoints_33[25],           # 13 LKnee
            keypoints_33[27],           # 14 LAnkle
            keypoints_33[2],            # 15 REye
            keypoints_33[5],            # 16 LEye
            keypoints_33[8],            # 17 REar
            keypoints_33[7],            # 18 LEar
            keypoints_33[32],           # 19 LBigToe
            keypoints_33[31],           # 20 LSmallToe
            keypoints_33[29],           # 21 LHeel
            keypoints_33[28],           # 22 RBigToe
            keypoints_33[27],           # 23 RSmallToe
            keypoints_33[30]            # 24 RHeel
        ]

        body25 = np.array(body25)
        return body25
    
    def draw_skeleton(self, image, skeleton_points, save_path):
        """
        Draw body 25-like skeleton on top of the original image (not black background).
        """
        # Make a copy of the original image to draw overlays
        output_image = image.copy()
        h, w, _ = output_image.shape

        # Define approximate BODY25-like connections (with spine added)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Right arm
            (1, 5), (5, 6), (6, 7),               # Left arm
            (1, 8), (8, 9), (9, 10), (10, 11),    # Right leg
            (8, 12), (12, 13), (13, 14),          # Left leg
            (0, 15), (15, 17),                    # Right head
            (0, 16), (16, 18),                    # Left head
            (1, 8)                                # Spine (neck → hip)
        ]

        # Drawing bones
        for start, end in connections:
            x1, y1, c1 = skeleton_points[start]
            x2, y2, c2 = skeleton_points[end]
            if c1 > 0 and c2 > 0:
                cv2.line(output_image, (int(x1 * w), int(y1 * h)),
                        (int(x2 * w), int(y2 * h)), (0, 0, 255), 2)

        # Drawing joints
        for i, (x, y, c) in enumerate(skeleton_points):
            if c > 0:
                cv2.circle(output_image, (int(x * w), int(y * h)), 5, (0, 255, 255), -1)
                cv2.putText(output_image, str(i), (int(x * w) + 4, int(y * h) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        
        if save_path:
            
            if output_image.dtype == np.float32 or output_image.dtype == np.float64:
                # Convrsion 0,1 range back to 0,255
                image_to_save = (output_image * 255).astype(np.uint8)
            else:
                # Image is already uint8
                image_to_save = output_image
            
            image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_to_save)
            print(f"Skeleton overlay saved to: {save_path}")
        else:
            cv2.imshow("Skeleton Overlay", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
