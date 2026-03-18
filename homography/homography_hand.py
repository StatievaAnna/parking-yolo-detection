import cv2
import numpy as np
import matplotlib.pyplot as plt

class PointSelector():
    def __init__(self, window_name='Select points'):
        self.window_name = window_name
        self.points = []
        self.image = None
        self.img_copy = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x,y])
            print(f"Point {len(self.points)}: ({x},{y})")
            cv2.circle(self.img_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.img_copy, str(len(self.points)), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.img_copy)

    def select_points(self, image):
        self.image = image
        self.img_copy = image.copy()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.img_copy)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("Select points. Put 'q' for finish.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        return np.array(self.points, dtype=np.float32)
    
def calculateHomography(perspective_video_path, bird_view_path):
    # perspective = cv2.imread(perspective_path)
    cap = cv2.VideoCapture(perspective_video_path)
    success, perspective = cap.read()
    bird_view = cv2.imread(bird_view_path)

    selector = PointSelector()

    print("Point selecting on perspective image:")
    perp_points = selector.select_points(perspective)

    selector = PointSelector()

    print("Points selecting on bird view image:")
    bird_view_points = selector.select_points(bird_view)

    H, mask = cv2.findHomography(perp_points, bird_view_points, cv2.RANSAC, 5.0)

    print(f'Homography matrix: {H}')

    height, width = bird_view.shape[:2]
    warped = cv2.warpPerspective(perspective, H, (width, height))
    
    plt.figure(figsize=(15,5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB))
    plt.title("Source image")
    plt.scatter(perp_points[:,0], perp_points[:,1], c='red', s=50)
    for i, (x,y) in enumerate(perp_points):
        plt.text(x+10, y-10, str(i+1), color='red', fontsize=12)
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(bird_view, cv2.COLOR_BGR2RGB))
    plt.title("Bird's view image")
    for i, (x,y) in enumerate(bird_view_points):
        plt.text(x+10, y-10, str(i+1), color='green', fontsize=12)

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Transformed image")

    plt.tight_layout()
    plt.show()

    np.save('homography_matrix_4.npy', H)

if __name__ == "main":
    flow_id = 1
    camera_id = 2
    flow_path = f"cvpipeline/crop_flows/{flow_id}/{camera_id}.mp4"
    top_view_path = "cvpipeline/CHAD/imgs/top_view.png"
    calculateHomography(flow_path, top_view_path)