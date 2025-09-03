"""vision 작업을 위한 핵심 연산 및 I/O 함수 모음.

이 모듈은 다음을 제공합니다:
1. 핵심 연산을 위한 네임스페이스: Camera_Process, Pose_Proc, Image_Proc.
2. 데이터 직렬화 및 파일 시스템 상호작용을 위한 클래스: Load_Proc, Save_Proc.
"""

import json
from pathlib import Path
from typing import Literal
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from .vision_types import (
    VEC_2D, IMG_SIZE, IN_M, VEC_3D, IMG_1C, TF_M, VEC_4D, IMG_3C, ROT_M
)
from .geometry import Convert
from .asset import Scene, Image, Point_Cloud, Camera

__all__ = [
    "Camera_Process", "Pose_Proc", "Image_Proc",
    "Load_Proc", "Save_Proc"
]


# ===================================================================
# Core Operation Classes
# ===================================================================

class Camera_Process:
    """카메라 모델 및 투영 계산을 위한 네임스페이스."""
    @staticmethod
    def Get_focal_length_from_fov(fov: VEC_2D, sz: IMG_SIZE) -> VEC_2D:
        return sz / (2 * np.tan(np.deg2rad(fov) / 2))

    @staticmethod
    def Get_size_from_fov(fov: VEC_2D, f_len: VEC_2D) -> IMG_SIZE:
        return (2 * np.round(np.tan(np.deg2rad(fov) / 2) * f_len)).astype(int)

    @staticmethod
    def Get_fov_from_size(sz: IMG_SIZE, f_len: VEC_2D) -> VEC_2D:
        return 2 * np.rad2deg(np.arctan2(sz / 2, f_len))

    @staticmethod
    def Get_principal_point(rate: VEC_2D, sz: IMG_SIZE) -> VEC_2D:
        return sz * rate

    @staticmethod
    def Get_principal_point_rate(pp: VEC_2D, sz: IMG_SIZE) -> VEC_2D:
        return pp / sz

    @staticmethod
    def Compose_intrinsic_matrix(f_len: VEC_2D, pp: VEC_2D) -> IN_M:
        if f_len.ndim != pp.ndim or f_len.shape[0] != pp.shape[0]:
            raise ValueError("Dimensions of f_len and pp do not match.")
        concat = np.concatenate([f_len, pp], axis=-1)
        if pp.ndim >= 2:
            in_m = np.tile(np.eye(3), [pp.shape[0], 1, 1])
            in_m[:, [0, 1, 0, 1], [0, 1, 2, 2]] = concat
        else:
            in_m = np.eye(3)
            in_m[[0, 1, 0, 1], [0, 1, 2, 2]] = concat
        return in_m

    @staticmethod
    def Extract_intrinsic_params(in_m: IN_M) -> tuple[VEC_2D, VEC_2D]:
        if in_m.ndim >= 3:
            p = in_m[:, [0, 1, 0, 1], [0, 1, 2, 2]]
            return p[:, :2], p[:, 2:]
        p = in_m[[0, 1, 0, 1], [0, 1, 2, 2]]
        return p[:2], p[2:]

    @staticmethod
    def Adjust_intrinsic_matrix(in_m: IN_M, sz: IMG_SIZE, new_sz: IMG_SIZE) -> IN_M:
        f_len, pp = Camera_Process.Extract_intrinsic_params(in_m)
        fov = Camera_Process.Get_fov_from_size(sz, f_len)
        pp_rate = Camera_Process.Get_principal_point_rate(pp, sz)
        new_f_len = Camera_Process.Get_focal_length_from_fov(fov, new_sz)
        new_pp = Camera_Process.Get_principal_point(pp_rate, new_sz)
        return Camera_Process.Compose_intrinsic_matrix(new_f_len, new_pp)

    @staticmethod
    def Apply_intrinsic_transform(pts: VEC_3D, in_m: IN_M, inv: bool = False) -> VEC_3D:
        mat = np.linalg.inv(in_m) if inv else in_m
        return (mat @ pts.T).T

    @staticmethod
    def Get_points_from_depth(d_map: IMG_1C, mask: IMG_1C | None = None) -> VEC_3D:
        h, w = d_map.shape[:2]
        m = mask.squeeze() if mask is not None else np.ones((h, w), dtype=bool)
        d = d_map[m].squeeze()
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        return np.c_[xx[m] * d, yy[m] * d, d]

    @staticmethod
    def Get_depth_map_from_points(pts: VEC_3D, d_sz: tuple[int, int]) -> IMG_1C:
        w, h = d_sz
        vis_pts = pts[pts[:, 2] > 1e-6]
        proj_pts = vis_pts.copy()
        proj_pts[:, :2] /= proj_pts[:, 2][:, None]
        u = np.round(proj_pts[:, 0]).astype(np.int32)
        v = np.round(proj_pts[:, 1]).astype(np.int32)
        m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, d = u[m], v[m], vis_pts[m, 2]
        d_flat = np.full(h * w, np.inf, dtype=np.float32)
        sort_idx = np.argsort(-d)
        v_w_u = v * w + u
        np.minimum.at(d_flat, v_w_u[sort_idx], d[sort_idx])
        d_flat[np.isinf(d_flat)] = 0
        return d_flat.reshape(h, w, 1)

    @staticmethod
    def Remap_depth_map(depth: IMG_1C, in_m: IN_M, new_sz: tuple[int, int]) -> IMG_1C:
        h, w = depth.shape[:2]
        new_in = Camera_Process.Adjust_intrinsic_matrix(in_m, np.array([w, h]), np.array(new_sz))
        pts_on_img = Camera_Process.Get_points_from_depth(depth)
        pts_on_cam = Camera_Process.Apply_intrinsic_transform(pts_on_img, in_m, inv=True)
        new_pts_on_img = Camera_Process.Apply_intrinsic_transform(pts_on_cam, new_in)
        return Camera_Process.Get_depth_map_from_points(new_pts_on_img, new_sz)

class Pose_Proc:
    """카메라 자세(extrinsic) 계산을 위한 네임스페이스."""
    @staticmethod
    def Compose_extrinsic_matrix(q: VEC_4D, t: VEC_3D) -> TF_M:
        mat = Convert.From_quat_to_matrix(q)
        if q.ndim >= 2:
            ext = np.tile(np.eye(4), [q.shape[0], 1, 1])
            ext[:, :3, :3], ext[:, :3, 3] = mat, t
        else:
            ext = np.eye(4)
            ext[:3, :3], ext[:3, 3] = mat, t
        return ext

    @staticmethod
    def Extract_extrinsic_params(ext: TF_M) -> tuple[VEC_4D, VEC_3D]:
        if ext.ndim >= 3:
            mat, t = ext[:, :3, :3], ext[:, :3, 3]
        else:
            mat, t = ext[:3, :3], ext[:3, 3]
        q = Convert.From_matrix_to_quat(mat)
        return q, t

    @staticmethod
    def Get_median_extrinsic(from_tfs: TF_M, to_tfs: TF_M) -> TF_M:
        assert len(from_tfs) == len(to_tfs)
        rel_tfs = to_tfs @ np.linalg.inv(from_tfs)
        tr = np.eye(4)
        r_vecs = R.from_matrix(rel_tfs[:, :3, :3]).as_rotvec()
        tr[:3, :3] = R.from_rotvec(r_vecs.mean(axis=0)).as_matrix()
        tr[:3, 3] = rel_tfs[:, :3, 3].mean(axis=0)
        return tr
    
    @staticmethod
    def Apply_extrinsic_transform(pts: VEC_3D | VEC_4D, ext: TF_M, inv: bool = False) -> VEC_3D:
        _ext = np.linalg.inv(ext) if inv else ext
        _ext = _ext[..., :3, :]
        pts_h = Convert.To_homogeneous(pts) if pts.shape[-1] == 3 else pts
        transformed_pts = np.einsum("...ij,...j->...i", _ext, pts_h)
        return transformed_pts

    @staticmethod
    def Remove_duplicate_poses(data: TF_M, precision: int) -> tuple[TF_M, np.ndarray]:
        scale: int = 10 ** precision
        flat_data = data.reshape(data.shape[0], -1)
        scaled = (flat_data * scale).round().astype(np.int64)
        _, indices = np.unique(scaled, axis=0, return_index=True)
        return data[indices], indices

CROP_MODE = Literal["pad", "crop_n", "crop_f", "crop_c"]

class Image_Proc:
    """이미지 처리 연산을 위한 네임스페이스."""
    @staticmethod
    def Get_new_shape(sz: tuple[int, int], ref_sz: int, unit: int = 14, by_w: bool = True, use_pad: bool = True) -> tuple[tuple[int, int], int]:
        w, h = sz
        ref = w if by_w else h
        other_dim = h if by_w else w
        rate = ref_sz / ref
        target_other_dim = round(other_dim * rate)
        dim_to_check = target_other_dim if by_w else ref_sz
        gap = (unit - (dim_to_check % unit)) % unit
        if not use_pad:
            gap = - (dim_to_check % unit) if dim_to_check % unit != 0 else 0
        new_sz_wh = (ref_sz, target_other_dim) if by_w else (target_other_dim, ref_sz)
        return new_sz_wh, gap

    @staticmethod
    def Adjust_size(img: IMG_3C | IMG_1C, mode: CROP_MODE, gap: int, is_w_dim: bool, fill: int | float = 0) -> np.ndarray:
        if gap == 0: return img
        abs_gap = abs(gap)
        if gap < 0:
            if mode == "crop_n": return img[:, :-abs_gap] if is_w_dim else img[:-abs_gap, :]
            if mode == "crop_f": return img[:, abs_gap:] if is_w_dim else img[abs_gap:, :]
            st, ed = abs_gap // 2, abs_gap - (abs_gap // 2)
            return img[:, st:-ed] if is_w_dim else img[st:-ed, :]
        st, ed = gap // 2, gap - (gap // 2)
        pad_v = ((0, 0), (st, ed), (0, 0)) if is_w_dim else ((st, ed), (0, 0), (0, 0))
        pad_dims = pad_v[:img.ndim]
        return np.pad(img, pad_dims, 'constant', constant_values=fill)

    @staticmethod
    def Resize_with_gap(img: IMG_3C | IMG_1C, sz: tuple[int, int], mode: CROP_MODE, gap: int, is_gap_on_w: bool) -> np.ndarray:
        r_img = cv2.resize(img, sz, interpolation=cv2.INTER_AREA)
        if r_img.ndim == 2: r_img = r_img[..., None]
        return Image_Proc.Adjust_size(r_img, mode, gap, is_gap_on_w)

    @staticmethod
    def Resize_image(img: IMG_3C | IMG_1C, mode: CROP_MODE, ref: int, unit: int, by_w: bool) -> np.ndarray:
        h, w = img.shape[:2]
        sz_wh, pad_gap = Image_Proc.Get_new_shape((w, h), ref, unit, by_w, mode == "pad")
        return Image_Proc.Resize_with_gap(img, sz_wh, mode, pad_gap, not by_w)

    @staticmethod
    def Visualize_image(img: IMG_1C, v_min: float | None = None, v_max: float | None = None, cmap=cv2.COLORMAP_JET, invalid_color=None) -> np.ndarray:
        if invalid_color is None: invalid_color = [0, 0, 0]
        valid_mask = (img > 0) & np.isfinite(img)
        valid_pixels = img[valid_mask]
        if v_min is None: v_min = np.min(valid_pixels) if len(valid_pixels) > 0 else 0
        if v_max is None: v_max = np.max(valid_pixels) if len(valid_pixels) > 0 else 1
        img_norm = np.clip((img - v_min) / max(v_max - v_min, 1e-8), 0, 1)
        img_u8 = (img_norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(img_u8.squeeze(), cmap)
        colored[~valid_mask.squeeze()] = invalid_color
        return colored

# ===================================================================
# Data Input/Output Process Classes
# ===================================================================

class Load_Proc:
    """에셋 및 씬 데이터 로딩을 위한 네임스페이스."""
    @staticmethod
    def Load_scene(file_path: Path) -> Scene:
        """JSON 파일로부터 Scene 객체를 로드합니다."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Scene.From_dict(data)

    @staticmethod
    def Load_image(image_asset: Image, data_root: Path):
        """Image asset의 이미지 데이터를 .data 필드로 로드합니다."""
        img_path = data_root / image_asset.relative_path
        if not img_path.exists():
            print(f"Warning: Image file not found at {img_path}")
            image_asset.data = None
            return
        bgr_img = cv2.imread(str(img_path))
        image_asset.data = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def Load_point_cloud(pc_asset: Point_Cloud, data_root: Path):
        """Point_Cloud asset의 포인트 클라우드 데이터를 로드합니다."""
        file_path = data_root / pc_asset.relative_path
        with np.load(file_path) as data:
            pc_asset.points = data['points']
            pc_asset.colors = data['colors']

    @staticmethod
    def Load_all_asset_data(scene: Scene, data_root: Path):
        """
        Scene 객체를 순회하며 모든 연관된 대용량 데이터를 메모리로 로드합니다.
        """
        print(f"Loading all asset data for scene '{scene.name}'...")
        # Scene-level 3D assets
        for asset in scene.assets.values():
            if isinstance(asset, Point_Cloud):
                Load_Proc.Load_point_cloud(asset, data_root)
        
        # Camera-level 2D assets
        for camera in scene.cameras.values():
            for asset in camera.assets.values():
                if isinstance(asset, Image):
                    Load_Proc.Load_image(asset, data_root)

class Save_Proc:
    """에셋 및 씬 데이터 저장을 위한 네임스페이스."""
    @staticmethod
    def Save_scene(scene: Scene, file_path: Path):
        """Scene 객체를 JSON 파일로 직렬화합니다."""
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(scene.To_dict(), f, indent=4)

    @staticmethod
    def Save_image(image_asset: Image, data_root: Path):
        """Image asset의 .data 필드에 있는 데이터를 파일로 저장합니다."""
        if image_asset.data is None: return
        img_path = data_root / image_asset.relative_path
        img_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(img_path), cv2.cvtColor(image_asset.data, cv2.COLOR_RGB2BGR))

    @staticmethod
    def Save_point_cloud(pc_asset: Point_Cloud, data_root: Path):
        """Point_Cloud asset의 데이터를 NPZ 파일로 저장합니다."""
        file_path = data_root / pc_asset.relative_path
        file_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(file_path, points=pc_asset.points, colors=pc_asset.colors)
    
    @staticmethod
    def Save_all_asset_data(scene: Scene, data_root: Path):
        """
        Scene 객체를 순회하며 메모리에 있는 모든 연관된 대용량 데이터를 파일로 저장합니다.
        """
        print(f"Saving all asset data for scene '{scene.name}'...")
        # Scene-level 3D assets
        for asset in scene.assets.values():
            if isinstance(asset, Point_Cloud):
                Save_Proc.Save_point_cloud(asset, data_root)
                
        # Camera-level 2D assets
        for camera in scene.cameras.values():
            for asset in camera.assets.values():
                if isinstance(asset, Image):
                    Save_Proc.Save_image(asset, data_root)

