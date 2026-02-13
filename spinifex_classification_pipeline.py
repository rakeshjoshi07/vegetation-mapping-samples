"""
Spinifex Classification Pipeline
Complete workflow: data preparation -> training -> inference
For vegetation fuel type mapping from high-resolution aerial imagery
"""

import cv2
import numpy as np
import os
import pickle
import sys
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
import rasterio.transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Missing dependencies. Install with: pip install segmentation-models-pytorch albumentations")
    sys.exit(1)

from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

USE_TEXTURE_ONLY = False
ROOT_DIR = 'spinifex_output/'

TRAINING_DATA_PAIRS = [
    {'image_path': 'training_bound_AOI_11_03_Sep2021_1.tif',
     'shapefile_path': 'training_labels_AOI_11_03_Sep2021_1.shp',
     'label_column': 'labels'},
    {'image_path': 'training_bound_AOI_11_03_Sep2021_2.tif',
     'shapefile_path': 'training_labels_AOI_11_03_Sep2021_2.shp',
     'label_column': 'labels'},
]

TRAIN_IMAGE_DIR = os.path.join(ROOT_DIR, 'train', 'images')
TRAIN_MASK_DIR = os.path.join(ROOT_DIR, 'train', 'masks')
VAL_IMAGE_DIR = os.path.join(ROOT_DIR, 'val', 'images')
VAL_MASK_DIR = os.path.join(ROOT_DIR, 'val', 'masks')
DATASET_STATS_FILE = os.path.join(ROOT_DIR, 'dataset_stats.pkl')
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'spinifex_model.pth')

TILE_SIZE = 256
OVERLAP_PERCENT = 0.5
VALIDATION_SPLIT = 0.2
MIN_POLYGON_AREA_RATIO = 0.3
MIN_MASK_PIXELS = 20
GENERATE_NEGATIVE_SAMPLES = True
NEGATIVE_SAMPLE_RATIO = 0.3

NUM_EPOCHS = 50
BATCH_SIZE = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 5
MODEL_ARCHITECTURE = 'unet'
ENCODER_NAME = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'


# =============================================================================
# Data Preparation Functions
# =============================================================================

class IncrementalStatistics:
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
        self.n = 0
        self.mean = np.zeros(num_channels, dtype=np.float64)
        self.M2 = np.zeros(num_channels, dtype=np.float64)
    
    def update(self, batch):
        batch = batch.reshape(-1, self.num_channels).astype(np.float64)
        for pixel in batch:
            self.n += 1
            delta = pixel - self.mean
            self.mean += delta / self.n
            delta2 = pixel - self.mean
            self.M2 += delta * delta2
    
    def get_statistics(self):
        if self.n < 2:
            return self.mean, np.ones(self.num_channels)
        variance = self.M2 / self.n
        std = np.sqrt(variance)
        std = np.where(std == 0, 1, std)
        return self.mean.astype(np.float32), std.astype(np.float32)


def get_gdal_patch(dataset, x_offset, y_offset, width, height):
    try:
        x_offset = max(0, min(x_offset, dataset.RasterXSize - 1))
        y_offset = max(0, min(y_offset, dataset.RasterYSize - 1))
        width = min(width, dataset.RasterXSize - x_offset)
        height = min(height, dataset.RasterYSize - y_offset)
        
        if width <= 0 or height <= 0:
            return None
        
        patch = dataset.ReadAsArray(x_offset, y_offset, width, height)
        if patch is None:
            return None
        
        if patch.ndim == 2:
            patch = np.stack([patch, patch, patch], axis=2)
        elif patch.ndim == 3 and patch.shape[0] < patch.shape[2]:
            patch = np.transpose(patch, (1, 2, 0))
        
        if patch.shape[2] > 3:
            patch = patch[:, :, :3]
        
        if patch.shape[0] < height or patch.shape[1] < width:
            padded = np.zeros((height, width, patch.shape[2]), dtype=patch.dtype)
            padded[:patch.shape[0], :patch.shape[1], :] = patch
            patch = padded
        
        patch = patch.astype(np.float32)
        
        if USE_TEXTURE_ONLY:
            gray = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
            patch = np.stack([gray, gray, gray], axis=2)
        
        return patch
    except Exception as e:
        print(f"Error reading patch at ({x_offset}, {y_offset}): {e}")
        return None


def calculate_combined_image_statistics(training_pairs, sample_ratio=0.01):
    print("Calculating image statistics...")
    stats = IncrementalStatistics(num_channels=3)
    
    for pair in training_pairs:
        image_path = pair['image_path']
        print(f"  Sampling from: {image_path}")
        
        if not os.path.exists(image_path):
            continue
        
        gdal_dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
        if gdal_dataset is None:
            continue
        
        img_width = gdal_dataset.RasterXSize
        img_height = gdal_dataset.RasterYSize
        num_samples = int(img_width * img_height * sample_ratio / (TILE_SIZE * TILE_SIZE))
        num_samples = max(50, min(num_samples, 500))
        
        for _ in range(num_samples):
            x = np.random.randint(0, max(1, img_width - TILE_SIZE))
            y = np.random.randint(0, max(1, img_height - TILE_SIZE))
            patch = get_gdal_patch(gdal_dataset, x, y, TILE_SIZE, TILE_SIZE)
            if patch is not None:
                stats.update(patch)
        
        gdal_dataset = None
    
    mean, std = stats.get_statistics()
    print(f"Mean: {mean}, Std: {std}")
    return mean, std


def create_robust_mask(polygons_in_tile_coords, tile_shape=(256, 256)):
    mask = np.zeros(tile_shape, dtype=np.uint8)
    
    for poly in polygons_in_tile_coords:
        try:
            if poly.is_empty:
                continue
            exterior_coords = np.array(poly.exterior.coords, dtype=np.int32)
            if len(exterior_coords) > 2:
                cv2.fillPoly(mask, [exterior_coords], color=255)
                for hole in poly.interiors:
                    hole_coords = np.array(hole.coords, dtype=np.int32)
                    if len(hole_coords) > 2:
                        cv2.fillPoly(mask, [hole_coords], color=0)
        except Exception as e:
            continue
    
    if np.sum(mask) > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask


def is_good_training_sample(image_patch, mask_patch):
    if np.std(image_patch) < 5:
        return False
    mask_pixels = np.sum(mask_patch > 0)
    if mask_pixels < MIN_MASK_PIXELS:
        return False
    mask_ratio = mask_pixels / (mask_patch.shape[0] * mask_patch.shape[1])
    if mask_ratio > 0.8:
        return False
    return True


def generate_training_tiles(image_path, shapefile_path, label_column, mean, std):
    print(f"Processing: {os.path.basename(image_path)}")
    
    gdal_dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if gdal_dataset is None:
        return
    
    img_width = gdal_dataset.RasterXSize
    img_height = gdal_dataset.RasterYSize
    gt = gdal_dataset.GetGeoTransform()
    proj = gdal_dataset.GetProjection()
    
    if not os.path.exists(shapefile_path):
        gdal_dataset = None
        return
    
    try:
        gdf = gpd.read_file(shapefile_path)
        from pyproj import CRS
        gdf_crs = CRS(gdf.crs)
        raster_crs = CRS.from_wkt(proj)
        
        if not gdf_crs.equals(raster_crs):
            gdf = gdf.to_crs(raster_crs)
        
        if label_column not in gdf.columns:
            gdal_dataset = None
            return
    except Exception as e:
        gdal_dataset = None
        return
    
    step_size = int(TILE_SIZE * (1 - OVERLAP_PERCENT))
    if step_size == 0:
        step_size = TILE_SIZE
    
    transform_matrix = rasterio.transform.Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
    inverse_transform = ~transform_matrix
    
    yielded = 0
    for y_offset_base in range(0, img_height, step_size):
        for x_offset_base in range(0, img_width, step_size):
            x_offset = min(x_offset_base, img_width - TILE_SIZE)
            y_offset = min(y_offset_base, img_height - TILE_SIZE)
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)
            
            tile_patch = get_gdal_patch(gdal_dataset, x_offset, y_offset, TILE_SIZE, TILE_SIZE)
            if tile_patch is None:
                continue
            
            tile_patch_normalized = (tile_patch - mean) / std
            tile_patch_8bit = np.clip((tile_patch_normalized * 64 + 128), 0, 255).astype(np.uint8)
            
            tile_min_lon, tile_max_lat = gdal.ApplyGeoTransform(gt, x_offset, y_offset)
            tile_max_lon, tile_min_lat = gdal.ApplyGeoTransform(gt, x_offset + TILE_SIZE, y_offset + TILE_SIZE)
            tile_geo_bbox = box(min(tile_min_lon, tile_max_lon), min(tile_min_lat, tile_max_lat),
                               max(tile_min_lon, tile_max_lon), max(tile_min_lat, tile_max_lat))
            
            intersecting_polygons = gdf[gdf.geometry.intersects(tile_geo_bbox)]
            
            mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
            has_features = False
            
            if not intersecting_polygons.empty:
                feature_polygons = []
                for poly_idx, poly_row in intersecting_polygons.iterrows():
                    poly_geom = poly_row.geometry
                    poly_label = poly_row[label_column]
                    
                    if poly_label == 1:
                        def transform_point(x_geo, y_geo):
                            x_img, y_img = inverse_transform * (x_geo, y_geo)
                            return x_img - x_offset, y_img - y_offset
                        
                        transformed_poly = shapely_transform(transform_point, poly_geom)
                        if not transformed_poly.is_valid:
                            transformed_poly = transformed_poly.buffer(0)
                        
                        tile_bounds = box(0, 0, TILE_SIZE, TILE_SIZE)
                        clipped_poly = transformed_poly.intersection(tile_bounds)
                        
                        if not clipped_poly.is_empty:
                            area_ratio = clipped_poly.area / poly_geom.area if poly_geom.area > 0 else 0
                            if area_ratio > MIN_POLYGON_AREA_RATIO:
                                if clipped_poly.geom_type == 'Polygon':
                                    feature_polygons.append(clipped_poly)
                                elif clipped_poly.geom_type == 'MultiPolygon':
                                    feature_polygons.extend([p for p in clipped_poly.geoms if p.geom_type == 'Polygon'])
                
                if feature_polygons:
                    mask = create_robust_mask(feature_polygons, (TILE_SIZE, TILE_SIZE))
                    if np.sum(mask > 0) >= MIN_MASK_PIXELS:
                        has_features = True
            
            if is_good_training_sample(tile_patch_8bit, mask):
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                tile_id = f"{image_name}_tile_{x_offset}_{y_offset}.png"
                yielded += 1
                yield {
                    'tile_id': tile_id,
                    'image_patch': tile_patch_8bit,
                    'mask': mask,
                    'has_features': has_features
                }
    
    gdal_dataset = None
    print(f"  Generated {yielded} tiles")


def prepare_training_data():
    print("="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    if os.path.exists(ROOT_DIR):
        import shutil
        shutil.rmtree(ROOT_DIR)
    
    for dir_path in [TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, VAL_IMAGE_DIR, VAL_MASK_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    valid_pairs = [p for p in TRAINING_DATA_PAIRS if os.path.exists(p['image_path']) and os.path.exists(p['shapefile_path'])]
    if not valid_pairs:
        print("No valid training data found!")
        sys.exit(1)
    
    print(f"Found {len(valid_pairs)} training pairs")
    mean, std = calculate_combined_image_statistics(valid_pairs)
    
    all_tile_ids = {'positive': [], 'negative': []}
    for pair in valid_pairs:
        for tile_data in generate_training_tiles(pair['image_path'], pair['shapefile_path'], 
                                                 pair['label_column'], mean, std):
            tile_id = tile_data['tile_id']
            try:
                cv2.imwrite(os.path.join(TRAIN_IMAGE_DIR, tile_id), tile_data['image_patch'])
                cv2.imwrite(os.path.join(TRAIN_MASK_DIR, tile_id), tile_data['mask'])
                category = 'positive' if tile_data['has_features'] else 'negative'
                all_tile_ids[category].append(tile_id)
            except:
                continue
    
    positive_ids = all_tile_ids['positive']
    negative_ids = all_tile_ids['negative']
    
    if GENERATE_NEGATIVE_SAMPLES and negative_ids and positive_ids:
        desired_negatives = int(len(positive_ids) * NEGATIVE_SAMPLE_RATIO)
        if len(negative_ids) > desired_negatives:
            negative_ids = list(np.random.choice(negative_ids, desired_negatives, replace=False))
    
    all_ids = positive_ids + negative_ids
    has_features = [True] * len(positive_ids) + [False] * len(negative_ids)
    
    print(f"\nTotal samples: {len(all_ids)} ({len(positive_ids)} positive, {len(negative_ids)} negative)")
    
    train_ids, val_ids = train_test_split(all_ids, test_size=VALIDATION_SPLIT, 
                                          stratify=has_features, random_state=42)
    
    for tile_id in val_ids:
        try:
            src_img = os.path.join(TRAIN_IMAGE_DIR, tile_id)
            src_mask = os.path.join(TRAIN_MASK_DIR, tile_id)
            if os.path.exists(src_img) and os.path.exists(src_mask):
                os.rename(src_img, os.path.join(VAL_IMAGE_DIR, tile_id))
                os.rename(src_mask, os.path.join(VAL_MASK_DIR, tile_id))
        except:
            continue
    
    dataset_info = {
        'mean': mean, 'std': std,
        'num_train_samples': len(train_ids),
        'num_val_samples': len(val_ids),
        'tile_size': TILE_SIZE,
        'training_sources': [os.path.basename(p['image_path']) for p in valid_pairs]
    }
    
    with open(DATASET_STATS_FILE, 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print(f"Training: {len(train_ids)}, Validation: {len(val_ids)}")
    print("="*60)


# =============================================================================
# Training Functions
# =============================================================================

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        self.image_files = [f for f in self.image_files 
                           if os.path.exists(os.path.join(mask_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = cv2.imread(os.path.join(self.image_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask.float()


def get_transforms(is_training=True, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        bce_loss = self.bce(predictions, targets.float())
        dice_loss = self.dice(predictions, targets.float())
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(predictions, targets, threshold=0.5):
    predictions = torch.sigmoid(predictions)
    pred_binary = (predictions > threshold).float()
    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = targets.view(-1).cpu().numpy()
    iou = jaccard_score(target_flat, pred_flat, average='binary', zero_division=0)
    f1 = f1_score(target_flat, pred_flat, average='binary', zero_division=0)
    return iou, f1


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = total_iou = total_f1 = 0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        iou, f1 = calculate_metrics(outputs, masks)
        total_iou += iou
        total_f1 += f1
    
    return (total_loss / len(dataloader), 
            total_iou / len(dataloader), 
            total_f1 / len(dataloader))


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = total_iou = total_f1 = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            iou, f1 = calculate_metrics(outputs, masks)
            total_iou += iou
            total_f1 += f1
    
    return (total_loss / len(dataloader), 
            total_iou / len(dataloader), 
            total_f1 / len(dataloader))


def train_model():
    print("="*60)
    print("TRAINING")
    print("="*60)
    
    if not os.path.exists(DATASET_STATS_FILE):
        print("Dataset stats not found. Run data preparation first.")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    with open(DATASET_STATS_FILE, 'rb') as f:
        dataset_stats = pickle.load(f)
    
    mean = dataset_stats.get('mean', [128, 128, 128])
    std = dataset_stats.get('std', [64, 64, 64])
    
    if isinstance(mean, np.ndarray):
        mean = mean.tolist()
    if isinstance(std, np.ndarray):
        std = std.tolist()
    if max(mean) > 1:
        mean = [m/255.0 for m in mean]
    if max(std) > 1:
        std = [s/255.0 for s in std]
    
    train_dataset = SegmentationDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, 
                                       get_transforms(True, mean, std))
    val_dataset = SegmentationDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, 
                                     get_transforms(False, mean, std))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
                    in_channels=3, classes=1, activation=None)
    model = model.to(device)
    
    print(f"Model: {MODEL_ARCHITECTURE} ({ENCODER_NAME})")
    
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_iou = 0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_iou, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou, val_f1 = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_iou)
        
        print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'model_config': {
                    'architecture': MODEL_ARCHITECTURE,
                    'encoder_name': ENCODER_NAME,
                    'encoder_weights': ENCODER_WEIGHTS
                },
                'dataset_stats': dataset_stats
            }, MODEL_SAVE_PATH)
            print(f"✓ Best model saved! Val IoU: {val_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without improvement")
            break
    
    print(f"\nTraining complete! Best Val IoU: {best_val_iou:.4f}")
    print("="*60)


# =============================================================================
# Inference Functions
# =============================================================================

def generate_tile_positions(img_width, img_height, tile_size, overlap_percent):
    step_size = int(tile_size * (1 - overlap_percent))
    if step_size == 0:
        step_size = tile_size
    
    positions = []
    for y_offset_base in range(0, img_height, step_size):
        for x_offset_base in range(0, img_width, step_size):
            x_offset = min(x_offset_base, img_width - tile_size)
            y_offset = min(y_offset_base, img_height - tile_size)
            positions.append((max(0, x_offset), max(0, y_offset)))
    return positions


def process_tiles_in_batches(gdal_dataset, positions, transform, model, device, 
                             img_width, img_height, batch_size=32, use_tta=True):
    total_batches = (len(positions) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Processing"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(positions))
        batch_positions = positions[start_idx:end_idx]
        
        batch_tiles = []
        valid_positions = []
        
        for x_offset, y_offset in batch_positions:
            tile_patch = get_gdal_patch(gdal_dataset, x_offset, y_offset, TILE_SIZE, TILE_SIZE)
            if tile_patch is not None:
                batch_tiles.append(tile_patch)
                valid_positions.append((x_offset, y_offset))
        
        if not batch_tiles:
            continue
        
        batch_transformed = [transform(image=tile.astype(np.uint8))['image'] for tile in batch_tiles]
        batch_tensor = torch.stack(batch_transformed).to(device)
        
        if use_tta:
            with torch.no_grad():
                pred = torch.sigmoid(model(batch_tensor))
                pred_hflip = torch.sigmoid(model(torch.flip(batch_tensor, dims=[3])))
                pred_hflip = torch.flip(pred_hflip, dims=[3])
                pred_vflip = torch.sigmoid(model(torch.flip(batch_tensor, dims=[2])))
                pred_vflip = torch.flip(pred_vflip, dims=[2])
                predictions = (pred + pred_hflip + pred_vflip) / 3
        else:
            with torch.no_grad():
                predictions = torch.sigmoid(model(batch_tensor))
        
        predictions_np = predictions.cpu().numpy()
        for pred, pos in zip(predictions_np, valid_positions):
            yield pred[0], pos


def stitch_predictions_weighted(prediction_generator, img_width, img_height, tile_size):
    result = np.zeros((img_height, img_width), dtype=np.float32)
    weight_sum = np.zeros((img_height, img_width), dtype=np.float32)
    
    y_coords, x_coords = np.ogrid[:tile_size, :tile_size]
    center_y, center_x = tile_size / 2, tile_size / 2
    sigma = tile_size / 4
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    
    for pred, (x_offset, y_offset) in prediction_generator:
        x_end = min(x_offset + tile_size, img_width)
        y_end = min(y_offset + tile_size, img_height)
        pred_h = y_end - y_offset
        pred_w = x_end - x_offset
        
        result[y_offset:y_end, x_offset:x_end] += pred[:pred_h, :pred_w] * weights[:pred_h, :pred_w]
        weight_sum[y_offset:y_end, x_offset:x_end] += weights[:pred_h, :pred_w]
    
    result = np.divide(result, weight_sum, out=np.zeros_like(result), where=weight_sum > 0)
    return result


def save_geotiff(output_path, data, reference_dataset, dtype=gdal.GDT_Byte):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_path, data.shape[1], data.shape[0], 1, dtype,
                               options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    out_dataset.SetGeoTransform(reference_dataset.GetGeoTransform())
    out_dataset.SetProjection(reference_dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(data)
    out_band.SetNoDataValue(0)
    out_band.FlushCache()
    out_dataset = None
    print(f"Saved: {output_path}")


def run_inference(inference_image_path, output_path, batch_size=32, threshold=0.5, use_tta=True):
    print("="*60)
    print("INFERENCE")
    print("="*60)
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Model not found. Train first.")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    dataset_stats = checkpoint['dataset_stats']
    
    model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=None,
                    in_channels=3, classes=1, activation=None)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    gdal_dataset = gdal.Open(inference_image_path, gdal.GA_ReadOnly)
    if gdal_dataset is None:
        print("Cannot open image")
        sys.exit(1)
    
    img_width = gdal_dataset.RasterXSize
    img_height = gdal_dataset.RasterYSize
    print(f"Image: {img_width} x {img_height}")
    
    mean = dataset_stats.get('mean', [128, 128, 128])
    std = dataset_stats.get('std', [64, 64, 64])
    if isinstance(mean, np.ndarray):
        mean = mean.tolist()
    if isinstance(std, np.ndarray):
        std = std.tolist()
    if max(mean) > 1:
        mean = [m/255.0 for m in mean]
    if max(std) > 1:
        std = [s/255.0 for s in std]
    
    transform = get_transforms(False, mean, std)
    positions = generate_tile_positions(img_width, img_height, TILE_SIZE, OVERLAP_PERCENT)
    print(f"Processing {len(positions):,} tiles...")
    
    prediction_generator = process_tiles_in_batches(gdal_dataset, positions, transform, 
                                                    model, device, img_width, img_height, 
                                                    batch_size, use_tta)
    final_prediction = stitch_predictions_weighted(prediction_generator, img_width, img_height, TILE_SIZE)
    
    binary_prediction = (final_prediction > threshold).astype(np.uint8) * 255
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_geotiff(output_path, binary_prediction, gdal_dataset)
    
    prob_output = output_path.replace('.tif', '_probability.tif')
    prob_map = (final_prediction * 255).astype(np.uint8)
    save_geotiff(prob_output, prob_map, gdal_dataset)
    
    gdal_dataset = None
    
    positive_pixels = np.sum(binary_prediction > 0)
    coverage = (positive_pixels / binary_prediction.size) * 100
    print(f"Coverage: {coverage:.2f}%")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Spinifex Classification Pipeline')
    parser.add_argument('--mode', choices=['prepare', 'train', 'inference', 'all'], 
                       default='all', help='Pipeline mode')
    parser.add_argument('--inference_image', type=str, help='Path to inference image')
    parser.add_argument('--output', type=str, help='Output prediction path')
    args = parser.parse_args()
    
    if args.mode in ['prepare', 'all']:
        prepare_training_data()
    
    if args.mode in ['train', 'all']:
        train_model()
    
    if args.mode == 'inference' or (args.mode == 'all' and args.inference_image):
        if not args.inference_image or not args.output:
            print("Inference requires --inference_image and --output")
            sys.exit(1)
        run_inference(args.inference_image, args.output)
