import os
import json

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_places_on_top():
    perspective_path = "cvpipeline/crop_flows/1/4.mp4"
    bird_view_path = "cvpipeline/CHAD/imgs/top_view.png"

    cap = cv2.VideoCapture(perspective_path)
    success, perspective = cap.read()

    bird_view = cv2.imread(bird_view_path)

    H = np.load("homography_matrix_4.npy", allow_pickle=True)
    height, width = bird_view.shape[:2]
    warped = cv2.warpPerspective(perspective, H, (width, height))

    map = np.load("map.npy", allow_pickle=True).item()

    plt.figure(figsize=(15,5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB))
    plt.title("Source image")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(bird_view, cv2.COLOR_BGR2RGB))
    plt.title("Bird's view image")
    for i in map.keys():
        x = [point[0] for point in map[i]]
        y = [point[1] for point in map[i]]
        plt.scatter(x, y, c='red', s=50)
        # x_m = (map[i][1][0] - map[i][3][0]) // 2
        # y_m = (map[i][0][1] - map[i][2][0]) // 2
        # plt.text(x_m, y_m, str(i+1), color='red', fontsize=12)

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Transformed image")
    for i in map.keys():
        x = [point[0] for point in map[i]]
        y = [point[1] for point in map[i]]
        plt.scatter(x, y, c='red', s=50)
        # x_m = (map[i][1][0] - map[i][3][0]) // 2
        # y_m = (map[i][0][1] - map[i][2][0]) // 2
        # plt.text(x_m, y_m, str(i+1), color='red', fontsize=12)

    plt.tight_layout()
    plt.show()

def show_places_on_perspective(perspective_path, bird_view_path, H_path, map_path):

    cap = cv2.VideoCapture(perspective_path)
    success, perspective = cap.read()

    bird_view = cv2.imread(bird_view_path)

    H = np.load(H_path, allow_pickle=True)
    H_inv = np.linalg.inv(H)

    map = np.load(map_path, allow_pickle=True).item()

    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(bird_view, cv2.COLOR_BGR2RGB))
    plt.title("Bird view")

    for i in map.keys():
        points_bird = map[i]
    
        x = [point[0] for point in points_bird]
        y = [point[1] for point in points_bird]

        plt.scatter(x, y, c='red', s=10)

        x_m = (x[1] + x[2]) // 2
        y_m = (y[0] + y[1]) // 2

        plt.text(x_m - 20, y_m, str(i+1), color='red', fontsize=8)

    ax = plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB))
    plt.title("Perspective view")

    height, width = perspective.shape[:2]

    for i in map.keys():
        points_bird = map[i]
        points_bird = np.array(points_bird, dtype=np.float32).reshape(-1,1,2)
        points_persp = cv2.perspectiveTransform(points_bird, H_inv)
        points_persp = points_persp.reshape(-1, 2)
        relevant_point = 0
        for j, point in enumerate(points_persp):
            x = point[0]
            y = point[1]
            if  0 <= x <= width and 0 <= y <= height:
                relevant_point += 1
            # else:
            #     points_persp[j][0] = np.clip(x, 0,  width)
            #     points_persp[j][1] = np.clip(y, 0, height)
        if relevant_point == 0: continue
    
        x = [point[0] for point in points_persp]
        y = [point[1] for point in points_persp]

        plt.scatter(x, y, c='red', s=10)

        # rect_points = np.concatenate([x, y], axis=1)

        polygon = patches.Polygon(
            points_persp,
            closed=True,
            edgecolor="green",
            facecolor="blue",
            alpha=0.3,
            linewidth=2
        )
        ax.add_patch(polygon)

        x_m = sum(x) // 4
        y_m = sum(y) // 4

        plt.text(x_m - 20, y_m, str(i+1), color='red', fontsize=8)

    plt.tight_layout()
    plt.show()


# perspective_path = "videos/4_001_0.mp4"
# bird_view_path = "CHAD/imgs/top_view.png"
# H_path = "homography_matrix_4.npy"
# map_path = "map.npy"

# show_places_on_perspective(perspective_path, bird_view_path, H_path, map_path)

class RectConstructor():
    def __init__(self, camera_id, flow_id, save_path, C=0.5):
        self.cfg = self.load_stream_config()

        map_path = self.cfg.get("map_path", "")
        self.top_map = np.load(map_path, allow_pickle=True).item()

        H_path = self.cfg.get('H_path', "") + f"/camera_id.npy"
        self.H = np.load(H_path)

        camera_points_path = self.cfg.get("camera_points_path", "")
        self.camera_points = np.load(camera_points_path, allow_pickle=True).item()
        
        self.lines_points = []
        self.window_name = "Construst spots"
        self.unit_length = None
        self.C = C
        self.vp = None
        self.H_inv = np.linalg.inv(self.H)
        self.camera_id = camera_id
        self.places_3d = {}
        self.save_points_path = self.cfg.get("places_3d_points_path", "")
        self.save_masks_3d_path = self.cfg.get("masks_3d_path", "")
        self.save_masks_2d_path = self.cfg.get("masks_2d_path", "")
        self.flow_id = flow_id
        self.video_path = self.cfg.get("flow_paths", "") + f"/{flow_id}/{camera_id}.mp4"
        self.img_copy = cv2.VideoCapture()

    def load_stream_config():
        config_path = "cvpipeline/config/stream_config.yaml"
        cfg = json.load(config_path)
        return cfg


    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.lines_points.append((x, y))
            cv2.circle(self.img_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(self.window_name, self.img_copy)

    def select_points(self):

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.img_copy)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("Select points. Put 'q' for finish.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        return self.lines_points
    
    def find_vanish_point(self):

        lines_points = self.select_points()

        (x1,y1) = lines_points[0]
        (x2,y2) = lines_points[1]
        (x3,y3) = lines_points[2]
        (x4,y4) = lines_points[3]

        # y1 = ax1 + b
        # y2 = ax2 + b

        a1 = (y1 - y2) / (x1 - x2)
        b1 = ((y1 + y2) -  a1 * (x1 + x2)) / 2

        a2 = (y3 - y4) / (x3 - x4)
        b2 = ((y3 + y4) -  a2 * (x3 + x4)) / 2

        # y = a1 * x + b1
        # y = a2 * x + b2

        x = (b2 - b1) / (a1 - a2) 
        y = (x * (a1 + a2) + (b1 + b2)) / 2

        self.vp = (x, y)

        return self.vp
    
    def set_unit_from_spot(self):
        """
        Устанавливает единицу длины по первому размеченному месту
        """
        spot_bird = self.top_map[0]
        # self.unit_length = np.linalg.norm(spot_bird[0] - spot_bird[1])
        self.unit_length = spot_bird[0][1] - spot_bird[1][1]
        print(f"1 единица = {self.unit_length:.1f} пикселей в bird-view")
        print(f"Это соответствует ~5 метрам в реальности")
    
    def get_distance_from_camera(self, place_id, point_id):
        """
        Возвращает расстояние от камеры до точки
        """

        camera_point = self.camera_points[self.camera_id]
        point_point = self.top_map[place_id][point_id]

        distance = np.linalg.norm(np.array(point_point) - np.array(camera_point))

        return distance / self.unit_length
    
    def calculate_k(self, place_id, point_id, car_height_units=0.3):

        d_units = self.get_distance_from_camera(place_id, point_id)

        k = self.C * car_height_units / max(0.1, d_units) 
        k = min(0.9, k)
        return k
        

    def compute_top_point(self, p_bottom, place_id, point_id,car_height_units=0.3):
        
        v = self.vp - p_bottom
        k = self.calculate_k(place_id, point_id, car_height_units=car_height_units)
        p_top = p_bottom - v * k
        
        return p_top
    
    def create_parallelepiped_mask_from_image(self, image_with_box, points_bottom, points_top):
        """
        Создает бинарную маску параллелепипеда по точкам
        points_bottom: 4 нижние точки
        points_top: 4 верхние точки
        """
        h, w = image_with_box.shape[:2]
        mask_3d = np.zeros((h, w), dtype=np.uint8)
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        
        faces = [
            np.array([points_bottom[0], points_bottom[1], 
                    points_bottom[2], points_bottom[3]]),
            
            np.array([points_top[0], points_top[1], 
                    points_top[2], points_top[3]]),
            
            np.array([points_bottom[0], points_bottom[1], 
                    points_top[1], points_top[0]]),
            np.array([points_bottom[1], points_bottom[2], 
                    points_top[2], points_top[1]]),
            np.array([points_bottom[2], points_bottom[3], 
                    points_top[3], points_top[2]]),
            np.array([points_bottom[3], points_bottom[0], 
                    points_top[0], points_top[3]])
        ]
        
        for face in faces:
            cv2.fillPoly(mask_3d, [np.int32(face)], 255)

        cv2.fillPoly(mask_2d, [np.int32(faces[0])], 255)
        
        return mask_3d, mask_2d
    
    def draw_spot_3d(self, place_id, car_height_units=0.3, color=None, save_results=False):
        """
        Рисует 3D параллелепипед для одного места
        """
        self.find_vanish_point()

        spot_bird = self.top_map[place_id-1]

        if self.unit_length is None:
            self.set_unit_from_spot()
        
        bottom = cv2.perspectiveTransform(
            np.array(spot_bird, dtype=np.float32).reshape(-1,1,2), self.H_inv).reshape(-1, 2)
        
        top = []
        for point_id, pt in enumerate(bottom):
            top_pt = self.compute_top_point(pt, place_id-1, point_id,  car_height_units)
            top.append(top_pt)
        top = np.array(top)

        self.places_3d[place_id] = {}

        self.places_3d[place_id]['bottom'] = bottom
        self.places_3d[place_id]['top'] = top

        mask_3d, mask_2d = self.create_parallelepiped_mask_from_image(self.img_copy, bottom, top)
        
        if save_results:
            save_path_masks_3d = self.save_masks_path + f"/masks_{self.camera_id+1}"
            filename = os.path.join(save_path_masks_3d, f"mask_{place_id}.png")
            cv2.imwrite(filename, mask_3d)

            save_path_masks_2d = self.save_masks_path + f"/masks_{self.camera_id+1}"
            filename = os.path.join(save_path_masks_2d, f"mask_{place_id}.png")
            cv2.imwrite(filename, mask_2d)

            save_points_path = self.save_points_path + f"/places_3d_{self.camera_id+1}.npy"
            filename = os.path.join(save_points_path)
            np.save(filename, self.places_3d)
        
        if color is None:
            color_bottom = (0, 255, 255)  # желтый
            color_top = (255, 255, 0)     # голубой
            color_vert = (0, 255, 0)       # зеленый
        else:
            color_bottom = color_top = color_vert = color
        
        cv2.polylines(self.img_copy, [np.int32(bottom)], True, color_bottom, 2)
        
        cv2.polylines(self.img_copy, [np.int32(top)], True, color_top, 2)
        
        for b, t in zip(bottom, top):
            cv2.line(self.img_copy, 
                    tuple(b.astype(int)), 
                    tuple(t.astype(int)), 
                    color_vert, 2)
        
        return self.img_copy
    
    def draw_all_3d_spots(self, save_results=False):
        for i in range(1,14):
            self.draw_spot_3d(i, save_results=save_results)
        
        return self.img_copy, self.places_3d


    def show_places(self, save_results=False):

        image_3d_place, places_3d = self.draw_all_3d_spots(save_results=save_results)

        cv2.imshow("Show 3d place", image_3d_place)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        return places_3d

if __name__ == "main":
        constructor = RectConstructor(camera_id=2, flow_id=1, C=0.37)

        constructor.show_places(save_masks=False) 
        # or
        # constructor.show_places(save_results=True, save_path_points=save_path_points)   




