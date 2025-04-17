from typing import Annotated

from enum import auto
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

import cv2

from python_ex.system import String


QUAT = Annotated[NDArray, (None, 4)]
RT = Annotated[NDArray, (None, 3, 3)]


class Convert():
    @staticmethod
    def M_to_Q(matrix: np.ndarray) -> QUAT:
        return R.from_matrix(matrix).as_quat(scalar_first=True)

    @staticmethod
    def Q_to_M(quat: np.ndarray) -> RT:
        return R.from_quat(quat, scalar_first=True).as_matrix()

    @staticmethod
    def M_2_Rv(matrix: np.ndarray) -> np.ndarray:
        return R.from_matrix(matrix).as_rotvec(degrees=True)

    @staticmethod
    def Rv_2_M(rotvec: np.ndarray) -> RT:
        return R.from_rotvec(rotvec, degrees=True).as_matrix()

    # @staticmethod
    # def Remove_duplicate(pts: ndarray, precision: int):
    #     _scale: int = 10 ** precision
    #     _scaled = (pts * _scale).round().astype(int64)  # 정수형 변환
    #     _, _indices = unique(_scaled, axis=0, return_index=True)

    #     return pts[_indices], _indices


TF = Annotated[NDArray, (None, 3, 1)]
TP = Annotated[NDArray, (None, 4, 4)]
INTRINSIC = Annotated[NDArray, (None, 3, 3)]

VEC_2 = Annotated[NDArray, (None, 2)]
VEC_3 = Annotated[NDArray, (None, 3)]
VEC_4 = Annotated[NDArray, (None, 4)]

IMG_SIZE = Annotated[NDArray[np.int32], (None, 2)]
PTS_COLOR = Annotated[NDArray[np.uint8], (None, 3)]


class Camera_Process():
    # about parameter
    @staticmethod
    def Get_focal_length_from(fov: VEC_2, size: VEC_2):
        return size / 2 * np.tan(fov / 2)

    @staticmethod
    def Get_size_from(fov: np.ndarray, f_length: np.ndarray):
        return 2 * np.round(np.tan(fov / 2) * f_length).astype(int)

    @staticmethod
    def Get_fov_from(size: np.ndarray, f_length: np.ndarray):
        return 2 * np.arctan2(size / 2, f_length)

    @staticmethod
    def Get_pp_from(rate: np.ndarray, size: np.ndarray):
        return size * rate

    @staticmethod
    def Get_pp_rate_from(pp: np.ndarray, size: np.ndarray):
        return pp / size

    # about intrinsic
    @staticmethod
    def Compose_intrinsic(f_length: np.ndarray, pp: np.ndarray):
        if f_length.ndim != pp.ndim or f_length.shape[0] != pp.shape[0]:
            raise ValueError

        _concat = np.concat([f_length, pp], axis=-1)

        if pp.ndim >= 2:
            _in = np.repeat(np.eye(3)[None, :, :], pp.shape[0], axis=0)
            _in[:, [0, 1, 0, 1], [0, 1, 2, 2]] = _concat

        else:
            _in = np.eye(3)
            _in[[0, 1, 0, 1], [0, 1, 2, 2]] = _concat

        return _in

    @staticmethod
    def Extract_intrinsic(intrinsic: INTRINSIC):
        if intrinsic.ndim >= 3:
            _param = intrinsic[:, [0, 1, 0, 1], [0, 1, 2, 2]]
            return _param[:, :2], _param[:, 2:]

        _param = intrinsic[[0, 1, 0, 1], [0, 1, 2, 2]]
        return _param[:2], _param[2:]

    @staticmethod
    def Adjust_intrinsic(
        intrinsic: INTRINSIC, size: IMG_SIZE,
        target: int | tuple[int, int] = 518,
        unit: int = 14, is_width: bool = True
    ) -> INTRINSIC:
        _ct_in = len(intrinsic)
        _ff, _pp = Camera_Process.Extract_intrinsic(intrinsic)
        _fov = Camera_Process.Get_fov_from(size, _ff)
        _pp_rate = Camera_Process.Get_pp_rate_from(_pp, size)

        if isinstance(target, tuple):
            _new_size = np.tile(target, [_ct_in, 1])

        else:
            _new_size = np.array(
                [
                    Image_Process.Get_new_shape(
                        _size, target, unit, is_width
                    ) for _size in size
                ],
                dtype=np.int32
            )
        # make new one
        return Camera_Process.Compose_intrinsic(
            Camera_Process.Get_focal_length_from(_fov, _new_size),
            Camera_Process.Get_pp_from(_pp_rate, _new_size)
        )

    # about extrinsic
    @staticmethod
    def Compose_extrinsic(tf: np.ndarray, q: np.ndarray):
        if (q.ndim != tf.ndim) or (q.ndim >= 2 and q.shape[0] != tf.shape[0]):
            raise ValueError

        _m = Convert.Q_to_M(q)

        if q.ndim >= 2:
            _ex = np.repeat(np.eye(4)[None, :, :], q.shape[0], axis=0)
            _ex[:, :3, :3] = _m
            _ex[:, :3, 3] = tf

        else:
            _ex = np.eye(4)
            _ex[:3, :3] = _m
            _ex[:3, 3] = tf

        return _ex

    @staticmethod
    def Extract_extrinsic(extrinsic: np.ndarray):
        if extrinsic.ndim >= 3:
            _q = Convert.M_to_Q(extrinsic[:, :3, :3])
            _tr = extrinsic[:, :3, 3]

        else:
            _q = Convert.M_to_Q(extrinsic[:3, :3])
            _tr = extrinsic[:3, 3]

        return _q, _tr

    @staticmethod
    def Get_median_extrinsic(
        from_tp: TP, to_tp: TP
    ) -> Annotated[NDArray, (4, 4)]:
        assert len(from_tp) == len(to_tp)

        _tp: TP = np.matmul(to_tp, np.linalg.inv(from_tp))
        _tr = np.eye(4)

        _tr[:3, :3] = Convert.Rv_2_M(
            Convert.M_2_Rv(_tp[:, :3, :3]).mean(axis=0))
        _tr[:3, 3] = _tp[:, :3, 3].mean(axis=0)

        return _tr


IMG_1C_GROUP = Annotated[NDArray, (None, None, None, 1)]
IMG_3C_GROUP = Annotated[NDArray, (None, None, None, 3)]

IMG_1C = Annotated[NDArray, (None, None, 1)]
IMG_3C = Annotated[NDArray, (None, None, 3)]


class Image_Process():
    @staticmethod
    def Get_new_shape(
        size: tuple[int, int], target_size: int,
        unit: int = 14, is_width: bool = True
    ) -> tuple[tuple[int, int], int]:
        _h, _w = size
        if is_width:
            _new_w = target_size
            _new_h = round(_h * (_new_w / _w))
            _pad_gap = _new_h % unit
        else:
            _new_h = target_size
            _new_w = round(_w * (_new_h / _h))
            _pad_gap = _new_w % unit

        return (_new_h, _new_w), _pad_gap

    class CROP(String.String_Enum):
        PAD = auto()
        CROP_C = auto()
        CROP_N = auto()
        CROP_F = auto()

    @staticmethod
    def Adjust_axis(
        img: IMG_1C | IMG_3C,
        mode: CROP, gap: int,
        unit: int = 14, is_width: bool = True
    ) -> np.ndarray:
        if mode == Image_Process.CROP.CROP_N:
            return img[:, :-gap] if is_width else img[:-gap, :]
        if mode == Image_Process.CROP.CROP_F:
            return img[:, gap:] if is_width else img[gap:, :]

        _st = gap // 2

        if mode == Image_Process.CROP.CROP_C:
            _ed = gap - _st
            return img[:, _st:-_ed] if is_width else img[_st:-_ed, :]

        _ed = unit - _st
        if is_width:
            _pad_v = ((0, 0), (_st, _ed), (0, 0))
        else:
            _pad_v = ((_st, _ed), (0, 0), (0, 0))

        return np.pad(img, _pad_v, constant_values=1.0)

    @staticmethod
    def Resize_img(
        image: IMG_1C | IMG_3C, mode: CROP,
        target: int | tuple[int, int] = 518,
        unit: int = 14, is_width: bool = True
    ) -> IMG_1C | IMG_3C:
        if isinstance(target, tuple):
            return cv2.resize(image, (target[1], target[0]))

        _size, _pad_gap = Image_Process.Get_new_shape(
            image.shape[:2], target, unit, is_width
        )

        _r_img = cv2.resize(image, (_size[1], _size[0]))

        if _pad_gap:
            return Image_Process.Adjust_axis(
                _r_img, mode, _pad_gap, unit, not is_width)

        return _r_img

    @staticmethod
    def Resize_depth_map(
        depth: IMG_1C,
        intrinsic: Annotated[NDArray, (3, 3)],
        new_intrinsic: Annotated[NDArray, (3, 3)],
        target: int | tuple[int, int] = 518,
        unit: int = 14, is_width: bool = True
    ):
        _pts_om_img = Point_process.Get_pts_from(depth)
        _pts_on_cam = Point_process.Apply_intrinsic(
            _pts_om_img, intrinsic, True)

        if isinstance(target, tuple):
            # make new intrinsic
            _size, _pad_gap = target, 0
        else:
            _size, _pad_gap = Image_Process.Get_new_shape(
                depth.shape[:2], target, unit, is_width
            )

        _new_d = Point_process.Get_depth_map_from(
            Point_process.Apply_intrinsic(_pts_on_cam, new_intrinsic),
            _size
        )

        if _pad_gap:
            _crop = Image_Process.CROP.CROP_N
            return Image_Process.Adjust_axis(
                _new_d, _crop, _pad_gap, unit, not is_width)

        return _new_d

    @staticmethod
    def Visualize_depth_map(
        _d: IMG_1C,
        threshold: tuple[float, float] | None = None, cmap=cv2.COLORMAP_JET
    ) -> np.ndarray:
        _d = np.nan_to_num(_d, nan=0.0, posinf=0.0, neginf=0.0)
        _d_8u: Annotated[NDArray[np.uint8], (None, None)] = np.round(
            ((_d - _d.min()) / (_d.max() - _d.min())) * 255
        ).astype(np.uint8)[:, :, 0]

        _colored_d = cv2.applyColorMap(_d_8u, cmap)

        if threshold:
            _th: Annotated[NDArray, (2)] = np.round((
                (np.array(threshold) - _d.min()) / (_d.max() - _d.min())
            ) * 255).astype(np.uint8)

            _mask = _d_8u >= _th.min() & _d_8u < _th.max()
            _colored_d[_mask] = [0, 0, 0]

        return _colored_d


class Point_process():
    # apply_transpose
    @staticmethod
    def Apply_extrinsic(
        pts: VEC_3 | VEC_4, extrinsic: Annotated[NDArray, (3, 4)],
        apply_inv: bool = False
    ) -> VEC_3:
        _pts: Annotated[NDArray, (4, None)] = np.c_[
            pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)
        ].T if pts.shape[1] == 3 else pts.T

        _ex = np.linalg.inv(extrinsic) if apply_inv else extrinsic

        return (_ex[:3, :] @ _pts)

    @staticmethod
    def Apply_intrinsic(
        pts: VEC_3, intrinsic: Annotated[NDArray, (3, 3)],
        apply_inv: bool = False
    ) -> VEC_3:
        return (np.linalg.inv(intrinsic) if apply_inv else intrinsic) @ pts.T

    @staticmethod
    def Get_pts_from(depth_map: IMG_1C, mask: IMG_1C | None = None):
        _d = depth_map[mask] if mask else depth_map  # activate depth
        _h, _w = depth_map.shape

        _yy, _xx = np.meshgrid(np.arange(_h), np.arange(_w), indexing='ij')
        return np.c_[_xx[mask] * _d, _yy[mask] * _d, _d]

    @staticmethod
    def Get_depth_map_from(
        pts: VEC_3, depth_map_size: tuple[int, int]
    ) -> IMG_1C:
        _h, _w = depth_map_size
        pts[:, :2] /= pts[:, 2]

        _u, _v = np.round(pts[:, :1]).astype(np.int32).T
        _mask = (_u >= 0 & _u < _w & _v >= 0 & _v < _h)

        _d_flat = np.full(_h * _w, np.inf, dtype=np.float32)
        np.minimum.at(
            _d_flat,
            (_v[_mask] * _w +  _u[_mask]).astype(np.int32),
            pts[_mask, 2]
        )
        _d_flat = np.nan_to_num(_d_flat, nan=0.0, posinf=0.0, neginf=0.0)
        return _d_flat.reshape(_h, _w)[:, :, None]

    @staticmethod
    def Remove_duplicate(
        data: TP, precision: int
    ):
        _scale: int = 10 ** precision
        _scaled = (data * _scale).round().astype(np.int64)  # 정수형 변환
        _, _indices = np.unique(_scaled, axis=0, return_index=True)

        return data[_indices], _indices
