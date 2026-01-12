import argparse
import os
import sys
import glob
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

# CUDA backend config (match demo settings)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Ensure project root is in sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pycolmap
import trimesh
import time
from PIL import Image
from torchvision import transforms as TF

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.eval_utils import (
    load_images_rgb,
)

def download_file_from_url(url, destination):
    import requests
    from tqdm import tqdm
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


def segment_sky(image_path, session):
    """
    return:
        output_mask: sky=0, other=255
        result_map_original: 0~255
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    result_map = run_skyseg(session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # 阈值32, threshold 32

    return output_mask, result_map_original

def run_skyseg(onnx_session, input_size, image):
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std 
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)  # 归一化到[0,1]
    onnx_result *= 255  
    return onnx_result.astype("uint8")


def apply_sky_segmentation(conf, image_path_list, input_sizes, output_path):
    S, H, W = conf.shape
    skyseg_model_path = "skyseg.onnx"

    if not os.path.exists(skyseg_model_path):
        print("Downloading skyseg.onnx...")
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", 
            skyseg_model_path
        )

    # 加载ONNX会话
    try:
        skyseg_session = onnxruntime.InferenceSession(skyseg_model_path)
    except Exception as e:
        print(f"无法加载天空分割模型: {e}")
        return conf

    sky_mask_list = []
    print("Generating sky masks (binary) and origin masks (grayscale)...")

    for i, (image_path, input_size) in enumerate(zip(image_path_list, input_sizes)):
        img_name = os.path.basename(image_path)
        
        sky_mask_dir = os.path.join(output_path, "sky_masks")
        sky_mask_path = os.path.join(sky_mask_dir, img_name)
        origin_mask_dir = os.path.join(output_path, "origin_mask")
        origin_mask_path = os.path.join(origin_mask_dir, img_name)

        binary_sky_mask, origin_gray_mask = segment_sky(image_path, skyseg_session)

        os.makedirs(sky_mask_dir, exist_ok=True)
        os.makedirs(origin_mask_dir, exist_ok=True)

        cv2.imwrite(sky_mask_path, binary_sky_mask)
        cv2.imwrite(origin_mask_path, origin_gray_mask)

        if binary_sky_mask.shape[0] != H or binary_sky_mask.shape[1] != W:
            binary_sky_mask = cv2.resize(binary_sky_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        sky_mask_list.append(binary_sky_mask)

    sky_mask_array = np.array(sky_mask_list)  # 形状 (S, H, W)
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)  # 非天空区域为1，天空为0
    updated_conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully:")
    print(f"  - Binary masks saved to: {os.path.join(output_path, 'sky_masks')}")
    print(f"  - Grayscale origin masks saved to: {os.path.join(output_path, 'origin_mask')}")
    return updated_conf

def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Build pycolmap camera params from intrinsics for different camera models.
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [
                intrinsics[fidx][0, 0],
                intrinsics[fidx][1, 1],
                intrinsics[fidx][0, 2],
                intrinsics[fidx][1, 2],
            ]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")
    return pycolmap_intri

def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_sizes,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Convert batched numpy arrays to a pycolmap.Reconstruction without building tracks.
    Only used to export an initialized reconstruction for visualization or as init.
    """
    N = len(extrinsics)
    P = len(points3d)

    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    for fidx in range(N):
        # 为每个图像创建独立的相机，处理不同尺寸
        pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)
        camera = pycolmap.Camera(
            model=camera_type,
            width=image_sizes[fidx][0],  # 使用当前图像的宽度
            height=image_sizes[fidx][1],  # 使用当前图像的高度
            params=pycolmap_intri,
            camera_id=fidx + 1,
        )
        reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )
        image = pycolmap.Image(
            id=fidx + 1,
            name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        points2D_list = []
        point2D_idx = 0
        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except Exception:
            print(f"frame {fidx + 1} does not have any points")
            image.registered = False

        reconstruction.add_image(image)

    return reconstruction

def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    input_sizes,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        img_coords = original_coords[pyimageid - 1]
        input_size = input_sizes[pyimageid - 1]
        
        real_width, real_height = img_coords[-2:]
        input_width, input_height = input_size
        
        width_ratio = real_width / input_width
        height_ratio = real_height / input_height
        
        pred_params = copy.deepcopy(pycamera.params)
        pred_params[0] *= width_ratio  # fx
        pred_params[1] *= height_ratio  # fy
        pred_params[2] *= width_ratio  # cx
        pred_params[3] *= height_ratio  # cy

        pycamera.params = pred_params
        pycamera.width = int(real_width)
        pycamera.height = int(real_height)

        if shift_point2d_to_original_res:
            top_left = img_coords[:2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * width_ratio

    return reconstruction


def run_vggt(model, vgg_input, dtype, image_paths=None):
    """
    Run VGGT to predict extrinsics, intrinsics, depth map and depth confidence.
    images: tensor [N, 3, H, W] in [0,1]
    """
    assert len(vgg_input.shape) == 4 and vgg_input.shape[1] == 3

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            vgg_input_cuda = vgg_input.cuda().to(torch.bfloat16)
            predictions = model(vgg_input_cuda, image_paths=image_paths)

    torch.cuda.synchronize()
    end = time.time()
    inference_time_ms = (end - start) * 1000.0

    print(
        f"VGGT inference time: {inference_time_ms:.1f} ms for {vgg_input.shape[0]} images"
    )
    # Measure max GPU VRAM usage
    if torch.cuda.is_available():
        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Max GPU VRAM used: {max_mem_mb:.2f} MB")

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], (vgg_input.shape[2], vgg_input.shape[3])
    )

    depth_tensor = predictions["depth"]
    depth_np = depth_tensor.detach().float().cpu().numpy()  # [N, H, W]
    depth_conf = predictions["depth_conf"]
    depth_conf_np = depth_conf.detach().float().cpu().numpy()  # [N, H, W]

    extrinsic_np = extrinsic.detach().float().cpu().numpy()  # [N, 4, 4]
    intrinsic_np = intrinsic.detach().float().cpu().numpy()  # [N, 3, 3]

    return extrinsic_np, intrinsic_np, depth_np, depth_conf_np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export COLMAP reconstruction from images using VGGT" 
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Dataset root containing images/ directory",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="./colmap_export_custom",
        help="Output directory (will create sparse/, sky_masks/, origin_mask/ under this directory)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/root/autodl-tmp/vggt-main/weights/model.pt",
        help="Model checkpoint file path",
    )
    parser.add_argument("--merging", type=int, default=0, help="Merging parameter")
    parser.add_argument(
        "--merge_ratio",
        type=float,
        default=0.9,
        help="Token merge ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=1,
        help="Depth confidence threshold to filter low-confidence depth",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=100000,
        help="Max number of 3D points to keep when exporting",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save the output images",
    )
    parser.add_argument(
        "--disable_sky_mask",
        action="store_true",
        help="Disable sky segmentation mask",
    )
    return parser.parse_args()


def load_and_preprocess_images(image_path_list, mode="pad"):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []
    input_sizes = []
    to_tensor = TF.ToTensor()
    target_size = 518

    for image_path in image_path_list:
        img = Image.open(image_path)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = (round(height * (new_width / width) / 14)) * 14
            else:
                new_height = target_size
                new_width = (round(width * (new_height / height) / 14)) * 14
        else:  # mode == "crop"
            new_width = target_size
            new_height = (round(height * (new_width / width) / 14)) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img.crop((0, start_y, new_width, start_y + target_size))
            new_height = target_size

        img_tensor = to_tensor(img)
        images.append(img_tensor)

        # 存储原始图像信息：[x1, y1, x2, y2, original_width, original_height]
        original_coords.append(np.array([0, 0, new_width, new_height, width, height]))
        input_sizes.append((new_width, new_height))

    shapes = [(img.shape[1], img.shape[2]) for img in images]
    if len(set(shapes)) > 1:
        print(f"Warning: Found images with different shapes: {set(shapes)}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        
        padded_images = []
        padded_input_sizes = []
        for img, coords in zip(images, original_coords):
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # 使用白色padding
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
                
                coords[0] += pad_left  # x1
                coords[1] += pad_top   # y1
                coords[2] += pad_left  # x2
                coords[3] += pad_top   # y2
            
            padded_images.append(img)
            padded_input_sizes.append((max_width, max_height))
        images = padded_images
        input_sizes = padded_input_sizes

    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    return images, original_coords, input_sizes


def filter_invalid_points(points_3d, points_xyf, points_rgb):
    valid_mask = np.isfinite(points_3d).all(axis=1)
    
    invalid_count = len(points_3d) - np.sum(valid_mask)
    if invalid_count > 0:
        print(f"⚠️  Found {invalid_count} invalid points (NaN or inf), filtering them out")
    
    filtered_points_3d = points_3d[valid_mask]
    filtered_points_xyf = points_xyf[valid_mask]
    filtered_points_rgb = points_rgb[valid_mask]
    
    return filtered_points_3d, filtered_points_xyf, filtered_points_rgb


def main():
    args = parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    # Device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"Using depth confidence threshold: {args.depth_conf_thresh}")
    print(f"Sky mask enabled: {not args.disable_sky_mask}")
    print(f"Output directory: {args.output_path.absolute()}")

    images_dir = args.data_path
    if not images_dir.exists():
        print(f"❌ images directory not found: {images_dir}")
        return

    image_path_list = sorted(glob.glob(os.path.join(str(images_dir), "*")))
    if len(image_path_list) == 0:
        print(f"❌ No images found in {images_dir}")
        return
    base_image_path_list = [os.path.basename(p) for p in image_path_list]
    print(f"🖼️  Found {len(image_path_list)} images")

    # Load model
    print(f"🔄 Loading model: {args.ckpt_path}")
    model = VGGT(merging=args.merging, merge_ratio=args.merge_ratio, vis_attn_map=False)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    print("✅ Model loaded")

    print(f"🔄 Loading and preprocessing images...")
    vgg_input, original_coords, input_sizes = load_and_preprocess_images(image_path_list, mode="pad")
    
    # 计算patch尺寸
    _, grid_h, grid_w = vgg_input.shape[1:]
    patch_width = grid_w // 14
    patch_height = grid_h // 14
    print(f"📐 Image patch dimensions: {patch_width}x{patch_height}")
    print(f"📐 Input image dimensions: {grid_w}x{grid_h}")

    # Update attention layer patch dimensions in the model
    model.update_patch_dimensions(patch_width, patch_height)

    extrinsic, intrinsic, depth_map, depth_conf = run_vggt(
        model, vgg_input, dtype, base_image_path_list
    )

    # 移除多余的批次维度
    if extrinsic.shape[0] == 1:
        extrinsic = extrinsic[0]
    if intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]
    if depth_map.shape[0] == 1:
        depth_map = depth_map[0]
    if depth_conf.shape[0] == 1:
        depth_conf = depth_conf[0]

    # 应用天空掩膜处理
    if not args.disable_sky_mask:
        print(f"🔄 Applying sky segmentation...")
        # 确保depth_conf的形状正确，同时传入输出路径（核心修改）
        if len(depth_conf.shape) == 3:
            # 转换为字符串路径，避免Path对象拼接问题
            depth_conf = apply_sky_segmentation(depth_conf, image_path_list, input_sizes, str(args.output_path))
        else:
            print(f"⚠️  depth_conf shape {depth_conf.shape} is not compatible with sky segmentation")

    all_points_3d = []
    all_points_xyf = []
    all_points_rgb = []

    num_frames = extrinsic.shape[0]

    for frame_idx in range(num_frames):
        print(f"🔄 Processing frame {frame_idx+1}/{num_frames}")
        
        # 单帧深度图反投影
        frame_depth = depth_map[frame_idx]  # 形状：(H, W, 1)
        frame_extrinsic = extrinsic[frame_idx]  # 形状：(3, 4)
        frame_intrinsic = intrinsic[frame_idx]  # 形状：(3, 3)
        
        # 1. 应用置信度过滤
        frame_conf_mask = depth_conf[frame_idx] >= args.depth_conf_thresh
        frame_conf_mask = frame_conf_mask.astype(bool)
        
        # 2. 创建过滤后的深度图
        filtered_depth = frame_depth.copy()
        filtered_depth[~frame_conf_mask] = 0
        
        # 反投影到3D点云
        frame_depth_expanded = filtered_depth[np.newaxis, ...]
        frame_extrinsic_expanded = frame_extrinsic[np.newaxis, ...]
        frame_intrinsic_expanded = frame_intrinsic[np.newaxis, ...]
        
        all_frames_points_3d = unproject_depth_map_to_point_map(
            frame_depth_expanded, 
            frame_extrinsic_expanded, 
            frame_intrinsic_expanded
        )
        
        frame_points_3d = all_frames_points_3d[0]
        
        # 获取当前帧的高度和宽度
        height, width, _ = frame_points_3d.shape
        
        # 生成单帧颜色
        frame_rgb = vgg_input[frame_idx]  # 形状：(3, H, W)
        frame_rgb = (frame_rgb.cpu().numpy() * 255).astype(np.uint8)
        frame_rgb = frame_rgb.transpose(1, 2, 0)  # 形状：(H, W, 3)
        
        # 生成单帧像素坐标网格
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        frame_xyf = np.stack([x_coords, y_coords, np.full_like(x_coords, frame_idx)], axis=-1)
        
        # 应用过滤
        filtered_points_3d = frame_points_3d[frame_conf_mask]
        filtered_xyf = frame_xyf[frame_conf_mask]
        filtered_rgb = frame_rgb[frame_conf_mask]
        
        # 收集所有帧的结果
        all_points_3d.append(filtered_points_3d)
        all_points_xyf.append(filtered_xyf)
        all_points_rgb.append(filtered_rgb)

    # 合并所有帧的点云数据
    points_3d = np.concatenate(all_points_3d, axis=0)
    points_xyf = np.concatenate(all_points_xyf, axis=0)
    points_rgb = np.concatenate(all_points_rgb, axis=0)

    print(f"📊 Total points before filtering: {len(points_3d)}")

    # 过滤无效点
    points_3d, points_xyf, points_rgb = filter_invalid_points(points_3d, points_xyf, points_rgb)

    print(f"📊 Total points after filtering: {len(points_3d)}")

    # 随机下采样到最大点数
    if len(points_3d) > args.max_points:
        indices = np.random.choice(len(points_3d), args.max_points, replace=False)
        points_3d = points_3d[indices]
        points_xyf = points_xyf[indices]
        points_rgb = points_rgb[indices]
        print(f"📊 Total points after downsampling: {len(points_3d)}")

    # 确保还有有效点
    if len(points_3d) == 0:
        print(f"❌ No valid points left after filtering, cannot generate PLY file")
        return

    # Build pycolmap reconstruction
    print("🧩 Converting to COLMAP format...")
    
    image_sizes = []
    for coords in original_coords:
        # coords: [x1, y1, x2, y2, original_width, original_height]
        input_width = coords[2] - coords[0]
        input_height = coords[3] - coords[1]
        image_sizes.append((int(input_width), int(input_height)))
    
    camera_type = "PINHOLE"
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_sizes,
        shared_camera=False,
        camera_type=camera_type,
    )

    # 准备输入尺寸列表
    input_sizes_for_rescale = [(grid_w, grid_h)] * num_frames  # 所有图像经过padding后尺寸相同
    
    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords,
        input_sizes_for_rescale,
        shift_point2d_to_original_res=False,
        shared_camera=False,
    )

    # Save
    sparse_dir = args.output_path / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving reconstruction to {sparse_dir.absolute()}")
    reconstruction.write(str(sparse_dir))

    # Quick point cloud PLY for visualization
    try:
        if points_rgb.shape[-1] == 3:
            points_rgb = np.clip(points_rgb, 0, 255).astype(np.uint8)
            point_cloud = trimesh.PointCloud(points_3d, colors=points_rgb)
            point_cloud.export(str(sparse_dir / "points.ply"))
            print(f"💾 Saved point cloud: {sparse_dir / 'points.ply'}")
        else:
            print(f"⚠️  Invalid RGB shape: {points_rgb.shape}, skipping PLY save")
    except Exception as e:
        print(f"⚠️  Failed to save PLY with trimesh: {e}")
        # 备选方案：使用numpy直接生成标准PLY文件
        try:
            ply_path = str(sparse_dir / "points.ply")
            with open(ply_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points_3d)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                for i in range(len(points_3d)):
                    x, y, z = points_3d[i]
                    r, g, b = points_rgb[i]
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        f.write(f"{x} {y} {z} {r} {g} {b}\n")
            print(f"💾 Saved PLY using numpy: {ply_path}")
        except Exception as e2:
            print(f"⚠️  Failed to save PLY with numpy: {e2}")

    print("🎉 Done.")


if __name__ == "__main__":
    with torch.no_grad():
        main()