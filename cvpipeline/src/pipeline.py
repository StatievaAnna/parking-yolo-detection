import os
import cv2
import json
import numpy as np
from pathlib import Path
from occuped_finder import OccupedFinder
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading

class SmartMultiCameraPlayer:
    def __init__(self, streams=None, target_size=(1920, 1080), output_video=None):
        self.stream_cfg = self.load_stream_config()
        self.streams = streams or self.stream_cfg.get("streams", {})
        self.target_size = target_size
        self.caps = {}
        self.fps_values = {}
        self.output_video = output_video
        self.video_writer = None
        self.raw_frames = {cam_id: None for cam_id in self.streams.keys()}
        self.frames_counter = 0
        self.lock = threading.Lock()
        
        self.detectors = {}
        self.places_3d_points = {}
        
        # Словарь для хранения голосов за занятые места
        self.occuped_places = {cam_id: [] for cam_id in self.streams.keys()} # for each camera
        self.votes = {}
        self.confirmed_occupied = set()
        self.model_cfg = self.load_model_config()
        self.place_ids = list(range(self.model_cfg["places"]["id_start"], self.model_cfg["places"]["id_end"] + 1))
        self.places_3d_points_path = self.stream_cfg.get("places_3d_points_path", "places_3d_points")
        self.map_path = self.stream_cfg.get("map_path", "map.npy")
        top_view_path = self.stream_cfg.get("top_view_path", "top_view.png")

        
        # Загрузка top-view изображения
        self.top_view = cv2.imread(top_view_path)
        if self.top_view is None:
            self.top_view = np.zeros((500, 500, 3), dtype=np.uint8)
            cv2.putText(self.top_view, "TOP VIEW", (150, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Загрузка координат мест на top-view
        try:
            self.top_view_places = np.load(self.map_path, allow_pickle=True).item()
        except Exception:
            print(f"⚠️ {self.map_path} не найден, создаю пустой словарь")
            self.top_view_places = {}
        
        # GPU настройки
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"✅ Используется устройство: {self.device}")
        
        self.use_gpu = torch.mps.is_available()
        if self.use_gpu:
            print(f"✅ GPU доступен: {torch.mps.is_available()}")
        else:
            print("⚠️ GPU не доступен, используется CPU")
        
        print(f"✅ Доступно ядер CPU: {mp.cpu_count()}")
        
        h, w = target_size
        
        # Сохраняем пропорции top-view
        tv_h, tv_w = self.top_view.shape[:2]
        tv_aspect = tv_w / tv_h
        
        # Рассчитываем размер top-view так, чтобы он занимал всю ширину
        self.top_view_display_width = w * 2
        self.top_view_display_height = int(self.top_view_display_width / tv_aspect)
        
        # Общая высота композита = 2 строки видео + top-view
        self.composite_height = h * 2 + self.top_view_display_height
        self.composite_width = w * 2
        
        self.composite = np.zeros((self.composite_height, self.composite_width, 3), dtype=np.uint8)
        
        print(f"📐 Размер композита: {self.composite_width} x {self.composite_height}")
        print(f"📐 Top-view: {self.top_view_display_width} x {self.top_view_display_height}")
        
        self.positions = {
            1: (0, 0),
            2: (w, 0),
            3: (0, h),
            4: (w, h)
        }
        
        for cam_id, flow_path in self.streams.items():
            video_flow = flow_path
            cap = cv2.VideoCapture(str(video_flow))

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 // 4)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 // 4)
            cap.set(cv2.CAP_PROP_FPS, 15) 
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            self.caps[cam_id] = cap
            self.fps_values[cam_id] = fps
            
            detector = OccupedFinder(cam_id, self.model_cfg, self.stream_cfg)
            detector.load_place_masks()
            
            self.detectors[cam_id] = detector
            
            print(f"Камера {cam_id}: FPS={fps:.2f}")

        self.target_fps = min(self.fps_values.values()) if self.fps_values else 15
        
        # Инициализируем VideoWriter
        if self.output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_video, 
                fourcc, 
                self.target_fps,
                (self.composite_width, self.composite_height)
            )
            print(f"📁 Запись видео в: {self.output_video}")
        
        # Создаем пул потоков
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def load_place_points(self):
        for camera_id in self.streams.keys():
            path = Path(self.places_3d_points_path) / f"places_3d_{camera_id}.npy"
            self.places_3d_points[camera_id] = np.load(path, allow_pickle=True).item()
            print(f"✅ Загружены places_3d_points для камеры {camera_id}")    
    
    def process_camera(self, cam_id, frame):
        """Обработка одной камеры"""
        detector = self.detectors[cam_id]
        occuped_places = detector.update_occuped_place(frame)
        return cam_id, occuped_places
    
    def update_votes(self, occuped_places_all):
        self.votes = {pid: occuped_places_all.get(pid, 0) for pid in self.place_ids}
        self.confirmed_occupied = {pid for pid, v in self.votes.items() if v >= 2}
    
    def draw_top_view(self):
        """Отрисовка top-view с подтвержденными занятыми местами"""
        # Масштабируем top-view до нужного размера с сохранением пропорций
        result = cv2.resize(self.top_view, (self.top_view_display_width, self.top_view_display_height))
        # Рисуем все места
        for place_id, points in self.top_view_places.items():
            place_id = place_id + 1
            color = (0, 255, 0)
            if place_id in self.confirmed_occupied:
                color = (0, 0, 255)
            
            # Масштабируем координаты под новый размер
            scale_x = self.top_view_display_width / self.top_view.shape[1]
            scale_y = self.top_view_display_height / self.top_view.shape[0]
            
            scaled_points = []
            for pt in points:
                x = int(pt[0] * scale_x)
                y = int(pt[1] * scale_y)
                scaled_points.append([x, y])
            
            pts = np.array(scaled_points, dtype=np.int32)
            cv2.polylines(result, [pts], True, color, 2)
            
            # Добавляем номер места
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(result, str(place_id), tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Добавляем информацию
        cv2.putText(result, f"Occupied: {len(self.confirmed_occupied)}/{len(self.place_ids)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result

    @staticmethod
    def load_stream_config():
        default_cfg = {
            "streams": {
                1: "rtsp://mediamtx:8554/cam1",
                2: "rtsp://mediamtx:8554/cam2",
                3: "rtsp://mediamtx:8554/cam3",
                4: "rtsp://mediamtx:8554/cam4",
            },
            "masks_3d_path": "prepeared_data/all_masks/masks_3d",
            "masks_2d_path": "prepeared_data/all_masks/masks_2d",
            "places_3d_points_path": "prepeared_data/all_points/places_3d_points",
            "map_path": "prepeared_data/all_points/map.npy",
            "camera_points_path": "prepeared_data/all_points/camera_points.npy",
        }
        config_path = Path(__file__).resolve().parents[1] / "config" / "stream_config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["streams"] = {int(k): v for k, v in cfg.get("streams", {}).items()}
            override_host = os.environ.get("PIPELINE_STREAM_HOST")
            if override_host:
                cfg["streams"] = {
                    cam_id: f"rtsp://{override_host}:8554/cam{cam_id}"
                    for cam_id in sorted(cfg["streams"].keys())
                }
            return cfg
        except Exception as e:
            print(f"stream_config.yaml не загружен ({e}), используются значения по умолчанию")
            return default_cfg

    @staticmethod
    def load_model_config():
        default_cfg = {
            "model": {"path": "yolov8n-seg.pt", "conf": 0.3, "classes": [2]},
            "places": {"id_start": 1, "id_end": 13, "mask_width": 640, "mask_height": 384, "mask_thresh": 127},
            "car": {"mask_thresh": 0.5, "min_area": 100},
            "matching": {"car_overlap": 0.8, "bottom_overlap": 0.2},
        }

        config_path = Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg
        except Exception as e:
            print(f"model_config.yaml не загружен ({e}), используются значения по умолчанию")
            return default_cfg
    
    def update_occuped_places(self):
        """Обновление занятых мест"""
        raw_frames = self.raw_frames.copy()
        occuped_places_all = {}

        # 1) создаем futures для каждого кадра
        futures = {
            self.thread_pool.submit(self.process_camera, cam_id, frame): cam_id
            for cam_id, frame in raw_frames.items() if frame is not None
        }

        # 2) агрегируем через as_completed
        for future in as_completed(futures):
            cam_id_res, occuped_places = future.result()
            for place_id in occuped_places:
                occuped_places_all[place_id] = occuped_places_all.get(place_id, 0) + 1
            with self.lock:
                self.occuped_places[cam_id_res] = occuped_places

        # 3) обновляем состояние автоматически
        with self.lock:
            self.update_votes(occuped_places_all)
    
    def get_frames_parallel(self):
        """Параллельное получение и обработка кадров"""
        self.frames_counter += 1

        for cam_id, cap in self.caps.items():
            ret, frame = cap.read()
            self.raw_frames[cam_id] = frame if ret else None
            if not ret:
                self._try_next_video(cam_id)

        if self.frames_counter % 60 == 0:  
            self.thread_pool.submit(self.update_occuped_places)
    
    def _try_next_video(self, cam_id):
        stream_path = self.streams.get(cam_id)
        if not stream_path:
            return
        print(f"🔄 Камера {cam_id}: переподключение потока")
        self.caps[cam_id].release()
        self.caps[cam_id] = cv2.VideoCapture(str(stream_path))
    
    def create_composite_with_topview(self, frames, top_view):
        """Сборка композита с сохранением пропорций"""
        h, w = self.target_size
        tv_h = self.top_view_display_height
        
        # Очищаем композит (черный фон)
        self.composite.fill(0)
        
        # Вставляем 4 видео в верхнюю часть с сохранением пропорций
        for cam_id, frame in frames.items():
            if frame is not None and cam_id in self.positions:
                x, y = self.positions[cam_id]
                
                # Ресайзим с сохранением пропорций, добавляем черные полосы если нужно
                frame_h, frame_w = frame.shape[:2]
                aspect = frame_w / frame_h
                target_aspect = w / h
                
                if aspect > target_aspect:
                    # Видео шире - добавляем черные полосы сверху/снизу
                    new_w = w
                    new_h = int(w / aspect)
                    resized = cv2.resize(frame, (new_w, new_h))
                    y_offset = y + (h - new_h) // 2
                    self.composite[y_offset:y_offset+new_h, x:x+new_w] = resized
                else:
                    # Видео выше - добавляем черные полосы слева/справа
                    new_h = h
                    new_w = int(h * aspect)
                    resized = cv2.resize(frame, (new_w, new_h))
                    x_offset = x + (w - new_w) // 2
                    self.composite[y:y+new_h, x_offset:x_offset+new_w] = resized
                
                # Добавляем номер камеры
                cv2.putText(self.composite, f"Cam {cam_id}", (x+10, y+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Вставляем top-view внизу (уже с правильными пропорциями)
        self.composite[h*2:h*2+tv_h, :] = top_view
        
        # Добавляем разделительную линию
        cv2.line(self.composite, (0, h*2), (w*2, h*2), (255, 255, 255), 2)
        
        return self.composite

    def drow_places_on_camera_view(self, frame, camera_id, occuped_places):
        for occuped_place_id in occuped_places:
            bottom = self.places_3d_points[camera_id][occuped_place_id]["bottom"]
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.int32(bottom)], (0, 0, 255))
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        for place_id in self.places_3d_points[camera_id].keys():
            bottom = self.places_3d_points[camera_id][place_id]["bottom"]
            top = self.places_3d_points[camera_id][place_id]["top"]

            color_bottom = (0, 255, 255)  # желтый
            color_top = (255, 255, 0)     # голубой
            color_vert = (0, 255, 0)      # зеленый
            
            cv2.polylines(frame, [np.int32(bottom)], True, color_bottom, 2)
            cv2.polylines(frame, [np.int32(top)], True, color_top, 2)
            
            for b, t in zip(bottom, top):
                cv2.line(frame, 
                        tuple(b.astype(int)), 
                        tuple(t.astype(int)), 
                        color_vert, 2) 

        return frame 

    def play(self):
        """Запуск плеера с записью видео"""
        print("\n" + "="*60)
        print("🎬 УМНЫЙ ПЛЕЕР С TOP-VIEW КАРТОЙ")
        print("="*60)
        print(f"Режим: {'GPU' if self.use_gpu else 'CPU'}")
        print(f"Обработка: ПАРАЛЛЕЛЬНАЯ")
        print(f"Размер видео: {self.composite_width} x {self.composite_height}")
        if self.output_video:
            print(f"📁 Запись в: {self.output_video}")
        print("Управление: ESC - выход, Пробел - пауза, P - переключить режим")
        print("="*60)
        
        cv2.namedWindow("Smart Parking Multi-Camera with Top View", cv2.WINDOW_NORMAL)
        
        frames_written = 0
        frames = {}

        self.load_place_points()  # Загружаем координаты мест для отрисовки на камерах
        
        try:
            while True:
                self.get_frames_parallel()
                
                if not any(f is not None for f in self.raw_frames.values()):
                    print("Все видео закончились")
                    break

                # Рисуем top-view
                top_view_with_places = self.draw_top_view()
                for camera_id, frame in self.raw_frames.items():
                    with self.lock:
                        occuped_places = list(self.occuped_places.get(camera_id, []))
                    if frame is not None:
                        frames[camera_id] = self.drow_places_on_camera_view(frame, camera_id, occuped_places)
                    else: 
                        frames[camera_id] = None

                # Создаем композит
                composite = self.create_composite_with_topview(frames, top_view_with_places)
                
                # Показываем
                cv2.imshow("Smart Parking Multi-Camera with Top View", composite)
                
                # Записываем
                if self.video_writer:
                    self.video_writer.write(composite)
                    frames_written += 1
                
                key = cv2.waitKey(10) & 0xFF
                if key == 27:
                    break
        
        finally:
            self.cleanup(frames_written)

    def cleanup(self, frames_written=0):
        print("\n👋 Завершение работы...")
        self.thread_pool.shutdown(wait=False)
        for cap in self.caps.values():
            cap.release()
        
        if self.video_writer:
            self.video_writer.release()
            print(f"📁 Видео сохранено: {self.output_video}")
            print(f"📊 Записано кадров: {frames_written}")
        
        cv2.destroyAllWindows()
        if self.use_gpu:
            torch.mps.empty_cache()

# Использование
if __name__ == "__main__":
    stream_cfg = SmartMultiCameraPlayer.load_stream_config()
    streams = stream_cfg.get("streams", {})
    
    player = SmartMultiCameraPlayer(
        streams=streams,
        target_size=(1080 // 4, 1920 // 4),
        output_video="cvpipeline/results_examples/parking_result_finish.mp4"
    )
    
    player.play()
