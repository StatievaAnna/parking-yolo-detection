from ultralytics import YOLO
import cv2
import numpy as np
import torch
from pathlib import Path

class OccupedFinder():
    def __init__(self, camera_id, model_cfg=None, stream_cfg=None):
        self.last_results = None
        self.last_masks = None
        self.last_frame_with_polygons = None
        self.occuped_places = []

        default_cfg = {
            "model": {"path": "yolov8n-seg.pt", "conf": 0.3, "classes": [2]},
            "places": {"id_start": 1, "id_end": 13, "mask_width": 640, "mask_height": 384, "mask_thresh": 127},
            "car": {"mask_thresh": 0.5, "min_area": 100},
            "matching": {"car_overlap": 0.8, "bottom_overlap": 0.2},
        }
        cfg = model_cfg or default_cfg
        stream_cfg = stream_cfg or {}

        model_part = cfg.get("model", {})
        places_part = cfg.get("places", {})
        car_part = cfg.get("car", {})
        match_part = cfg.get("matching", {})

        self.model_path = model_part.get("path", default_cfg["model"]["path"])
        self.conf = model_part.get("conf", default_cfg["model"]["conf"])
        self.classes = model_part.get("classes", default_cfg["model"]["classes"])

        self.place_start = places_part.get("id_start", default_cfg["places"]["id_start"])
        self.place_end = places_part.get("id_end", default_cfg["places"]["id_end"])
        self.mask_width = places_part.get("mask_width", default_cfg["places"]["mask_width"])
        self.mask_height = places_part.get("mask_height", default_cfg["places"]["mask_height"])
        self.mask_thresh = places_part.get("mask_thresh", default_cfg["places"]["mask_thresh"])

        self.car_mask_thresh = car_part.get("mask_thresh", default_cfg["car"]["mask_thresh"])
        self.car_min_area = car_part.get("min_area", default_cfg["car"]["min_area"])

        self.car_overlap_thresh = match_part.get("car_overlap", default_cfg["matching"]["car_overlap"])
        self.bottom_overlap_thresh = match_part.get("bottom_overlap", default_cfg["matching"]["bottom_overlap"])
        self.masks_3d_path = stream_cfg.get("masks_3d_path", "masks_3d")
        self.masks_2d_path = stream_cfg.get("masks_2d_path", "masks_2d")

        # MPS проверкаа
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"✅ Используется устройство: {self.device}")
        
        # Модель автоматически пойдет на GPU если доступно
        self.model = YOLO(self.model_path)
        self.camera_id = camera_id
        
        # Кэши на CPU (для совместимости)
        self.place_masks_binary_cache = {}
        self.place_masks_2d_binary_cache = {}
        self.place_area = {}
        self.bottom_area = {}
        
        # GPU-версии масок (для быстрых вычислений)
        self.place_masks_gpu = {}
        self.place_masks_2d_gpu = {}
        self.place_areas_gpu = {}
        self.bottom_areas_gpu = {}
        
        # Для пакетной обработки
        self.place_ids = list(range(self.place_start, self.place_end + 1))
        self.all_masks_gpu = None
        self.all_masks_2d_gpu = None
        self.all_areas_gpu = None
        self.all_bottom_areas_gpu = None   

    def load_place_masks(self):
        """Загрузка масок и создание GPU-копий с пакетной обработкой"""
        camera_id = self.camera_id
        print(f"\n📥 Загрузка масок для камеры {camera_id}...")

        masks_list = []
        masks_2d_list = []
        areas_list = []
        bottom_areas_list = []
        valid_place_ids = []

        for place_id in self.place_ids:
            # Загрузка 3D масок
            path = Path(self.masks_3d_path) / f"masks_{camera_id}" / f"mask_{place_id}.png"
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_resized = cv2.resize(mask, (self.mask_width, self.mask_height))
                mask_binary = (mask_resized > self.mask_thresh)
                self.place_masks_binary_cache[place_id] = mask_binary
                
                # GPU копия
                mask_gpu = torch.from_numpy(mask_binary).to(self.device)
                self.place_masks_gpu[place_id] = mask_gpu
                masks_list.append(mask_gpu)
                
                # Площадь
                area = mask_binary.sum()
                self.place_area[place_id] = area
                area_tensor = torch.tensor(area, device=self.device)
                self.place_areas_gpu[place_id] = area_tensor
                areas_list.append(area_tensor)

            # Загрузка 2D масок (bottom)
            path_2d = Path(self.masks_2d_path) / f"masks_{camera_id}" / f"mask_{place_id}.png"
            mask_2d = cv2.imread(str(path_2d), cv2.IMREAD_GRAYSCALE)
            if mask_2d is not None:
                mask_resized = cv2.resize(mask_2d, (self.mask_width, self.mask_height))
                mask_binary = (mask_resized > self.mask_thresh)
                self.place_masks_2d_binary_cache[place_id] = mask_binary
                
                # GPU копия
                mask_2d_gpu = torch.from_numpy(mask_binary).to(self.device)
                self.place_masks_2d_gpu[place_id] = mask_2d_gpu
                masks_2d_list.append(mask_2d_gpu)
                
                # Площадь bottom
                bottom_area = mask_binary.sum()
                self.bottom_area[place_id] = bottom_area
                bottom_area_tensor = torch.tensor(bottom_area, device=self.device)
                self.bottom_areas_gpu[place_id] = bottom_area_tensor
                bottom_areas_list.append(bottom_area_tensor)
            
            if place_id in self.place_masks_gpu:
                valid_place_ids.append(place_id)

        # Создаем пакетные тензоры для быстрой обработки
        if masks_list:
            self.all_masks_gpu = torch.stack(masks_list)  # [N, H, W]
            self.all_masks_2d_gpu = torch.stack(masks_2d_list)  # [N, H, W]
            self.all_areas_gpu = torch.tensor(areas_list, device=self.device)  # [N]
            self.all_bottom_areas_gpu = torch.tensor(bottom_areas_list, device=self.device)  # [N]
            self.valid_place_ids = valid_place_ids

        print(f"✅ Загружено {len(self.place_masks_gpu)} 3D масок и {len(self.place_masks_2d_gpu)} 2D масок")
        print(f"✅ Пакетные тензоры: all_masks_gpu.shape = {self.all_masks_gpu.shape if self.all_masks_gpu is not None else 'None'}")


    def check_intersections_batch(self, car_mask_binary_gpu, car_area_gpu):
        """Пакетная проверка пересечений со всеми местами сразу"""
        if self.all_masks_gpu is None:
            return []

        # Расширяем размерности для broadcasting
        car_mask_expanded = car_mask_binary_gpu.unsqueeze(0)  # [1, H, W]

        # Вычисляем пересечения со всеми масками сразу (GPU операция!)
        intersections = (car_mask_expanded & self.all_masks_gpu).sum(dim=(1, 2)).float()  # [N]
        intersections_bottom = (car_mask_expanded & self.all_masks_2d_gpu).sum(dim=(1, 2)).float()  # [N]
        
        # Условия
        condition1 = intersections / car_area_gpu > self.car_overlap_thresh
        condition2 = intersections_bottom / self.all_bottom_areas_gpu > self.bottom_overlap_thresh
        
        # Находим места, удовлетворяющие обоим условиям
        valid_mask = condition1 & condition2
        
        if valid_mask.any():
            # Вычисляем IoU для валидных мест
            unions = car_area_gpu + self.all_areas_gpu[valid_mask] - intersections[valid_mask]
            ious = intersections[valid_mask] / unions
            
            # Получаем индексы мест
            valid_indices = torch.where(valid_mask)[0]
            
            # Сортируем по IoU
            sorted_indices = torch.argsort(ious, descending=True)
            
            # Возвращаем список [place_id, iou]
            results = []
            for idx in sorted_indices:
                place_idx = valid_indices[idx].item()
                place_id = self.valid_place_ids[place_idx]
                iou_value = ious[idx].item()
                results.append([place_id, iou_value])
            
            return results
        
        return []

    def update_occuped_place(self, frame):
        
        occuped_places = []
    
        results = self.model(frame, conf=self.conf, classes=self.classes)
        self.last_results = results
        self.last_masks = results[0].masks if results and results[0].masks else None
    
        # 2. Пакетная обработка масок на GPU
        if self.last_masks is not None:
            masks_data = self.last_masks.data.cpu().numpy()
            
            # Обрабатываем все машины
            for i in range(len(masks_data)):
                car_mask = masks_data[i]
                
                # Маска на GPU
                car_mask_gpu = torch.from_numpy(car_mask).to(self.device)
                car_mask_binary_gpu = (car_mask_gpu > self.car_mask_thresh)
                car_area_gpu = torch.sum(car_mask_binary_gpu).float()
                
                if car_area_gpu < self.car_min_area:
                    continue
                
                # Пакетная проверка всех мест (один вызов GPU вместо 13!)
                ious = self.check_intersections_batch(car_mask_binary_gpu, car_area_gpu)
                
                # Закрашиваем занятое место
                if ious:
                    occuped_place_id = ious[0][0]  # лучшее совпадение
                    occuped_places.append(occuped_place_id)          

        return occuped_places
