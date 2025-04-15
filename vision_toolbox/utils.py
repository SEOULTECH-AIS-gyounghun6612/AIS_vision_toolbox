from enum import auto
import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2

from python_ex.system import String


# ROTATION = tuple[Literal["euler", "rovec", "quat"], list[float]]


class Transpose():
    @staticmethod
    def Matrix_2_quat(matrix: np.ndarray) -> np.ndarray:
        return R.from_matrix(matrix).as_quat(scalar_first=True)

    @staticmethod
    def Quat_2_matrix(quat: np.ndarray) -> np.ndarray:
        return R.from_quat(quat, scalar_first=True).as_matrix()

    @staticmethod
    def Matrix_2_rotvec(matrix: np.ndarray) -> np.ndarray:
        return R.from_matrix(matrix).as_rotvec(degrees=True)

    @staticmethod
    def Rotvec_2_matrix(rotvec: np.ndarray) -> np.ndarray:
        return R.from_rotvec(rotvec, degrees=True).as_matrix()

    @staticmethod
    def Get_intrinsic(k_value: tuple[float, float, float, float]):
        _in = np.eye(3)
        _in[[0, 1, 0, 1], [0, 1, 2, 2]] = k_value
        return _in

    @staticmethod
    def Get_k_value(intrinsic: np.ndarray):
        return intrinsic[:3, :3].reshape(-1)[[0, 4, 2, 5]]

    # @staticmethod
    # def Remove_duplicate(pts: ndarray, precision: int):
    #     _scale: int = 10 ** precision
    #     _scaled = (pts * _scale).round().astype(int64)  # 정수형 변환
    #     _, _indices = unique(_scaled, axis=0, return_index=True)

    #     return pts[_indices], _indices

    @staticmethod
    def Get_extrinsic(rotation: np.ndarray, transfer: np.ndarray):
        _ex = np.eye(4)
        _ex[:3, :3] = rotation
        _ex[:3, 3] = transfer

        return _ex

    @staticmethod
    def Pose_align(from_cam_tp: np.ndarray, to_cam_tp: np.ndarray):
        assert len(from_cam_tp) == len(to_cam_tp)

        _tp: np.ndarray = np.matmul(
            to_cam_tp, np.linalg.inv(from_cam_tp)
        )
        _align = np.eye(4)

        _align[:3, :3] = Transpose.Rotvec_2_matrix(
            Transpose.Matrix_2_rotvec(_tp[:, :3, :3]).mean(axis=0))
        _align[:3, 3] = _tp[:, :3, 3].mean(axis=0)

        return _align


class Image_process():
    @staticmethod
    def Fov_to_size(
        fov: tuple[float, float] | np.ndarray,
        f_length: tuple[float, float] | np.ndarray, is_degree: bool = True
    ):
        _fov = fov if isinstance(
            fov, np.ndarray
        ) else np.array(fov, dtype=np.float32)
        if is_degree:
            _fov *= np.pi / 180

        _f_length = f_length if isinstance(
            f_length, np.ndarray
        ) else np.array(f_length, dtype=np.float32)

        _half = np.round(np.tan(_fov / 2) * _f_length)
        return 2 * _half

    @staticmethod
    def Size_to_fov(size: tuple[int, int], f_length: tuple[float, float]):
        _size = size if isinstance(
            size, np.ndarray
        ) else np.array(size, dtype=np.float32)

        _f_length = f_length if isinstance(
            f_length, np.ndarray
        ) else np.array(f_length, dtype=np.float32)

        return 2 * np.arctan2(_size / 2, _f_length)

    @staticmethod
    def Focal_length_from_size_n_fov(
        fov: tuple[float, float] | np.ndarray,
        size: tuple[int, int] | np.ndarray, is_degree: bool = True
    ):
        _fov = fov if isinstance(
            fov, np.ndarray
        ) else np.array(fov, dtype=np.float32)
        if is_degree:
            _fov *= np.pi / 180

        _size = size if isinstance(
            size, np.ndarray
        ) else np.array(size, dtype=np.float32)

        return _size / (2 * np.tan(_fov / 2))

    @staticmethod
    def Get_resized_shape(
        size: tuple[int, int], target_size: int,
        unit: int = 14, is_width: bool = True
    ) -> tuple[int, int, int]:
        _h, _w = size
        if is_width:
            _new_w = target_size
            _new_h = round(_h * (_new_w / _w))
            _pad_gap = _new_h % unit
        else:
            _new_h = target_size
            _new_w = round(_w * (_new_h / _h))
            _pad_gap = _new_w % unit

        return _new_h, _new_w, _pad_gap

    class CROP(String.String_Enum):
        PAD = auto()
        CROP_C = auto()
        CROP_N = auto()
        CROP_F = auto()

    @staticmethod
    def Resize_img(
        image: np.ndarray,
        target_size: int = 518,
        unit: int = 14,
        is_width: bool = True
    ) -> tuple[np.ndarray, int]:
        _new_h, _new_w, _pad_gap = Image_process.Get_resized_shape(
            image.shape[:2], target_size, unit, is_width)
        _resized = cv2.resize(image, (_new_w, _new_h))
        return _resized, _pad_gap

    @staticmethod
    def Depth_to_points(
        depth: np.ndarray, mask: np.ndarray,
        k: tuple[float, float, float, float]  # fx, fy, cx, cy
    ) -> np.ndarray:
        _d = depth[mask] if depth.ndim == 3 else depth[mask, None]

        _H, _W = depth.shape
        _yy, _xx = np.meshgrid(np.arange(_H), np.arange(_W), indexing='ij')
        _pts = np.c_[_xx[mask, None] * _d, _yy[mask, None] * _d, _d]

        return _pts @ np.linalg.inv(Transpose.Get_intrinsic(k)).T

    @staticmethod
    def Resize_depth(
        depth: np.ndarray,
        k: tuple[float, float, float, float],  # fx, fy, cx, c
        target_size: int = 518,
        unit: int = 14,
        scale: float = 1000.,
        is_width: bool = True
    ) -> tuple[np.ndarray, tuple[float, float, float, float], int]:
        _new_h, _new_w, _pad_gap = Image_process.Get_resized_shape(
            depth.shape[:2], target_size, unit, is_width)

        _d = depth.astype(np.float32) / scale
        _d_mask = depth != 0

        _fov = Image_process.Size_to_fov(
            depth.shape[:2][::-1], k[:2])
        _n_fs = Image_process.Focal_length_from_size_n_fov(
            _fov, (_new_w, _new_h), False
        )

        _k = np.array(k)
        _new_k = np.r_[_n_fs, _n_fs * _k[2:] / _k[:2]]

        _new_i = Transpose.Get_intrinsic(_new_k)

        _pts = Image_process.Depth_to_points(_d, _d_mask, k)
        _new_pts = np.round((_pts @ _new_i.T) / _pts[:, 2, None])

        # 최소값을 위치별로 채우기
        _d_flat = np.full(_new_h * _new_w, np.inf, dtype=np.float32)
        np.minimum.at(
            _d_flat,
            (_new_pts[:, 1] * _new_w +  _new_pts[:, 0]).astype(np.int32),
            _pts[:, 2]
        )
        _d_flat[_d_flat == np.inf] = 0.0
        # 2D depth map으로 변환
        return (
            _d_flat.reshape(_new_h, _new_w)[:, :, None],
            np.r_[
                _fov, _new_k[:2] / _d_flat.shape[:2]
            ],
            _pad_gap
        )

    @staticmethod
    def Adjust_axis(
        img: np.ndarray, mode: CROP, gap: int,
        unit: int = 14, is_width: bool = True
    ) -> np.ndarray:
        if mode == Image_process.CROP.CROP_N:
            return img[:, :-gap] if is_width else img[:-gap, :]
        if mode == Image_process.CROP.CROP_F:
            return img[:, gap:] if is_width else img[gap:, :]

        _st = gap // 2

        if mode == Image_process.CROP.CROP_C:
            _ed = gap - _st
            return img[:, _st:-_ed] if is_width else img[_st:-_ed, :]

        _ed = unit - _st
        _pad_v = ((0, 0), (_st, _ed)) if is_width else ((_st, _ed), (0, 0))
        return np.pad(img, _pad_v, constant_values=1.0)


class Point_process():
    @staticmethod
    def Apply_Transform(
        points: np.ndarray,
        transforms: list[np.ndarray]
    ):
        # transform shape check
        _shapes = [_tr.shape[-2:] for _tr in transforms]
        assert _shapes[1:-1:2] == _shapes[2::2]

        # pts array shape check
        assert points.shape[-1] == _shapes[0]

        _pts = points.T
        for T in transforms:
            _pts = T @ _pts
        return _pts.T
