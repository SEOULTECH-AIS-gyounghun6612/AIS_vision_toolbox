from typing import Annotated, Literal

from enum import auto
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

import cv2

from python_ex.system  import String


RT = Annotated[NDArray, (None, 3, 3)]
TF = Annotated[NDArray, (None, 3, 1)]
TP = Annotated[NDArray, (None, 4, 4)]
INTRINSIC = Annotated[NDArray, (None, 3, 3)]

VEC_2 = Annotated[NDArray, (None, 2)]
VEC_3 = Annotated[NDArray, (None, 3)]
VEC_4 = Annotated[NDArray, (None, 4)]

IMGs_SIZE = Annotated[NDArray[np.int32], (None, 2)]
PTS_COLOR = Annotated[NDArray[np.uint8], (None, 3)]

IMG_1C_GROUP = Annotated[NDArray, (None, None, None, 1)]
IMG_3C_GROUP = Annotated[NDArray, (None, None, None, 3)]

IMG_1C = Annotated[NDArray, (None, None, 1)]
IMG_3C = Annotated[NDArray, (None, None, 3)]

L2R = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [-0.0, -0.0, -1.0, -0.0],
    [0.0, 0.0, 0.0, 1.0]
])

class Convert():
    class Rotation():
        """ ### 회전 변환 관련 기능을 제공하는 클래스

        행렬, 쿼터니언, 회전 벡터 간의 변환을 수행

        ---------------------------------------------------------------------------
        ### Structure
        - M_to_Q: 회전 행렬 → 쿼터니언 변환
        - Q_to_M: 쿼터니언 → 회전 행렬 변환
        - M_2_Rv: 회전 행렬 → 로드리게스 회전 벡터 변환
        - Rv_2_M: 로드리게스 회전 벡터 → 회전 행렬 변환
        """

        @staticmethod
        def M_to_Q(matrix: RT) -> VEC_4:
            """ ### 회전 행렬을 쿼터니언으로 변환

            ------------------------------------------------------------------
            ### Args
            - matrix: n개 3x3 회전 행렬 -> [n, 3, 3] 

            ### Returns
            - VEC_4: scalar-first 쿼터니언 (qw, qx, qy, qz) -> [n, 4]
            """
            return R.from_matrix(matrix).as_quat(scalar_first=True)

        @staticmethod
        def Q_to_M(quat: VEC_4) -> RT:
            """ ### 쿼터니언을 회전 행렬로 변환

            ------------------------------------------------------------------
            ### Args
            - quat: scalar-first 쿼터니언 (qw, qx, qy, qz) -> [n, 4]

            ### Returns
            - RT: n개 3x3 회전 행렬 -> [n, 3, 3] 
            """
            return R.from_quat(quat, scalar_first=True).as_matrix()

        @staticmethod
        def M_2_Rv(matrix: RT) -> VEC_3:
            """ ### 회전 행렬을 로드리게스 회전 벡터로 변환

            ------------------------------------------------------------------
            ### Args
            - matrix: n개 3x3 회전 행렬 -> [n, 3, 3] 

            ### Returns
            - VEC_3: degree 단위 로드리게스 회전 벡터 -> [n, 3] 
            """
            return R.from_matrix(matrix).as_rotvec(degrees=True)

        @staticmethod
        def Rv_2_M(rotvec: VEC_3) -> RT:
            """ ### 회전 벡터를 로드리게스 회전 행렬로 변환

            ------------------------------------------------------------------
            ### Args
            - rotvec: degree 단위 로드리게스 회전 벡터 -> [n, 3] 

            ### Returns
            - RT: n개 3x3 회전 행렬 -> [n, 3, 3] 
            """
            return R.from_rotvec(rotvec, degrees=True).as_matrix()

    @staticmethod
    def To_homog(obj: VEC_2 | VEC_3):
        return np.c_[
            obj, np.ones((obj.shape[0], 1), dtype=obj.dtype)
        ]


class Camera_Process():
    # about parameter
    @staticmethod
    def Get_focal_length_from(fov: VEC_2, size: IMGs_SIZE) -> VEC_2:
        return size / (2 * np.tan(fov / 2))

    @staticmethod
    def Get_size_from(fov: VEC_2, f_length: VEC_2) -> IMGs_SIZE:
        return 2 * np.round(np.tan(fov / 2) * f_length).astype(int)

    @staticmethod
    def Get_fov_from(size: IMGs_SIZE, f_length: VEC_2) -> VEC_2:
        return 2 * np.arctan2(size / 2, f_length)

    @staticmethod
    def Get_pp_from(rate: VEC_2, size: IMGs_SIZE) -> VEC_2:
        return size * rate

    @staticmethod
    def Get_pp_rate_from(pp: VEC_2, size: IMGs_SIZE) -> VEC_2:
        return pp / size

    # about intrinsic
    @staticmethod
    def Compose_intrinsic(f_length: VEC_2, pp: VEC_2) -> INTRINSIC:
        if f_length.ndim != pp.ndim or f_length.shape[0] != pp.shape[0]:
            raise ValueError

        _concat = np.concatenate([f_length, pp], axis=-1)

        if pp.ndim >= 2:
            _in = np.tile(np.eye(3), [pp.shape[0], 1, 1])
            _in[:, [0, 1, 0, 1], [0, 1, 2, 2]] = _concat

        else:
            _in = np.eye(3)
            _in[[0, 1, 0, 1], [0, 1, 2, 2]] = _concat

        return _in

    @staticmethod
    def Extract_intrinsic(intrinsic: INTRINSIC) -> tuple[VEC_2, VEC_2]:
        if intrinsic.ndim >= 3:
            _param = intrinsic[:, [0, 1, 0, 1], [0, 1, 2, 2]]
            return _param[:, :2], _param[:, 2:]

        _param = intrinsic[[0, 1, 0, 1], [0, 1, 2, 2]]
        return _param[:2], _param[2:]

    @staticmethod
    def Adjust_intrinsic(
        intrinsic: INTRINSIC, size: IMGs_SIZE, new_size: IMGs_SIZE
    ) -> INTRINSIC:
        _ff, _pp = Camera_Process.Extract_intrinsic(intrinsic)
        _fov = Camera_Process.Get_fov_from(size, _ff)
        _pp_rate = Camera_Process.Get_pp_rate_from(_pp, size)

        # make new one
        return Camera_Process.Compose_intrinsic(
            Camera_Process.Get_focal_length_from(_fov, new_size),
            Camera_Process.Get_pp_from(_pp_rate, new_size)
        )

    @staticmethod
    def Apply_intrinsic(
        pts: VEC_3, intrinsic: Annotated[NDArray, (3, 3)],
        apply_inv: bool = False
    ) -> VEC_3:
        _instrinsic = np.linalg.inv(intrinsic) if apply_inv else intrinsic
        return (_instrinsic @ pts.T).T

    # about transform in image plane to camera oridnate
    @staticmethod
    def Get_pts_from(depth_map: IMG_1C, mask: IMG_1C | None = None) -> VEC_3:
        _mask = mask if mask is not None else np.ones(
            depth_map.shape[:2], dtype=np.bool_)
        _d_map = depth_map[_mask, 0]
        _h, _w = depth_map.shape[:2]
        _yy, _xx = np.meshgrid(np.arange(_h), np.arange(_w), indexing='ij')

        return np.c_[_xx[_mask] * _d_map, _yy[_mask] * _d_map, _d_map]

    @staticmethod
    def Get_depth_map_from(
        pts: VEC_3, depth_map_size: tuple[int, int]
    ) -> IMG_1C:
        _w, _h = depth_map_size
        _vis_pts = pts[pts[:, 2] > 0]
        _vis_pts[:, :2] /= _vis_pts[:, 2][:, None]

        _u = np.round(_vis_pts[:, 0])
        _v = np.round(_vis_pts[:, 1])
        _mask = (
            (_u >= 0) * (_u < _w) & (_v >= 0) & (_v < _h))
        
        _u = _u[_mask].astype(np.int32)
        _v = _v[_mask].astype(np.int32)

        _d_flat = np.full(_h * _w, np.inf, dtype=np.float32)
        np.minimum.at(
            _d_flat,
            (_v * _w +  _u).astype(np.int32),
            _vis_pts[_mask, 2]
        )
        _d_flat = np.nan_to_num(_d_flat, nan=0.0, posinf=0.0, neginf=0.0)
        _d_flat[_d_flat < 0] = 0.0
        return _d_flat.reshape(_h, _w)[:, :, None]

    @staticmethod
    def Remapping_depth_map(
        depth: IMG_1C,
        intrinsic: Annotated[NDArray, (3, 3)],
        new_size: tuple[int, int]
    ):
        _h, _w = depth.shape[:2]
        _new_in = Camera_Process.Adjust_intrinsic(
            intrinsic,
            [_w, _h],
            new_size
        )

        _pts_om_img = Camera_Process.Get_pts_from(depth)
        _pts_on_cam = Camera_Process.Apply_intrinsic(
            _pts_om_img, intrinsic, True)

        return Camera_Process.Get_depth_map_from(
            Camera_Process.Apply_intrinsic(_pts_on_cam, _new_in),
            new_size
        )


class Image_Process():
    @staticmethod
    def Get_new_shape(
        size: tuple[int, int], ref_size: int, unit: int = 14,
        by_width: bool = True, use_paddig: bool = True
    ) -> tuple[tuple[int, int], int]:
        _ref = size[int(not by_width)]  # if by_width -> width else height
        _rate = ref_size / _ref

        _target = round(size[int(by_width)] * _rate)
        _gap = unit - (_target % unit) if use_paddig else - (_target % unit)

        if by_width:
            return (ref_size, _target), _gap
        return (_target, ref_size), _gap

    class CROP(String.String_Enum):
        PAD = auto()
        CROP_C = auto()
        CROP_N = auto()
        CROP_F = auto()

    @staticmethod
    def Adjust_size(
        img: IMG_1C | IMG_3C, mode: CROP, gap: int,
        is_width: bool = True, filling: int | float = 0
    ) -> np.ndarray:
        # if mode is crop_* -> gap is negative
        if mode == "crop_n":
            return img[:, :gap] if is_width else img[:gap, :]
        if mode == "crop_f":
            return img[:, -gap:] if is_width else img[-gap:, :]

        _gap = -gap if gap < 0 else gap
        _st = _gap // 2
        _ed = _gap - _st

        if mode == "crop_c":
            return img[:, _st:-_ed] if is_width else img[_st:-_ed, :]

        if is_width:
            _pad_v = ((0, 0), (_st, _ed), (0, 0))
        else:
            _pad_v = ((_st, _ed), (0, 0), (0, 0))

        return np.pad(img, _pad_v, constant_values=filling)

    @staticmethod
    def Resize_img_with_gap(
        image: IMG_1C | IMG_3C, size: tuple[int, int],
        mode: CROP, gap: int, is_width: bool = False
    ) -> IMG_1C | IMG_3C:
        _r_img = cv2.resize(image, size)

        if gap:
            return Image_Process.Adjust_size(_r_img, mode, gap, is_width, 255)

        return _r_img

    @staticmethod
    def Resize_img(
        image: IMG_1C | IMG_3C, mode: CROP,
        ref: int = 518,
        unit: int = 14, by_width: bool = True
    ) -> IMG_1C | IMG_3C:
        _size, _pad_gap = Image_Process.Get_new_shape(
            image.shape[-2::-1], ref, unit, by_width, mode == "pad"
        )

        return Image_Process.Resize_img_with_gap(
            image, _size, mode, _pad_gap, not by_width
        )

    @staticmethod
    def Visualize_image_with_threshold(
        image: IMG_1C,
        threshold: tuple[float, float] | None = None,
        cmap=cv2.COLORMAP_JET,
        visualize_ground: bool = False
    ) -> np.ndarray:
        if threshold:
            _h, _l = max(threshold), min(threshold)
            _mask = (image < _l) & (image > _h)
            image[_mask] = 0.0

        _min, _max = np.min(image), np.max(image)

        if _min == _max:
            _img = (np.ones_like(image) * 255).astype(np.uint8)
        else:
            _img = np.round(
                (image - _min) / (_max - _min) * 255).astype(np.uint8)

        _colored_d = cv2.applyColorMap(_img, cmap)

        if not visualize_ground:
            _colored_d[_img[..., 0] == 0] = [0, 0, 0]

        return _colored_d


class Space_process():
    @staticmethod
    def Hand_change(
        obj: VEC_3 | VEC_4 | TP, mode: Literal["L2R", "R2L"] = "L2R"
    ):
        assert mode == "L2R", print("mode: R2L is not yet" )
        _tp = L2R if mode == "L2R" else np.eye(4)

        _ndim = obj.ndim
        if _ndim == 2:  #  VEC_3 or VEC_4
            _ord = obj.shape[1]
            _pts: VEC_4 = Convert.To_homog(obj) if _ord == 3 else obj
            
            return Space_process.Apply_extrinsic(
                _pts, _tp[None, :, :])[0, :, :3]

        return _tp @ obj

    # about extrinsic
    @staticmethod
    def Compose_extrinsic(q: np.ndarray, tf: np.ndarray):
        if (q.ndim != tf.ndim) or (q.ndim >= 2 and q.shape[0] != tf.shape[0]):
            raise ValueError

        _m = Convert.Rotation.Q_to_M(q)

        if q.ndim >= 2:
            _ex = np.tile(np.eye(4), [q.shape[0], 1, 1])
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
            _q = Convert.Rotation.M_to_Q(extrinsic[:, :3, :3])
            _tr = extrinsic[:, :3, 3]

        else:
            _q = Convert.Rotation.M_to_Q(extrinsic[:3, :3])
            _tr = extrinsic[:3, 3]

        return _q, _tr

    @staticmethod
    def Get_median_extrinsic(
        from_tp: TP, to_tp: TP
    ) -> Annotated[NDArray, (4, 4)]:
        assert len(from_tp) == len(to_tp)

        _tp: TP = np.matmul(to_tp, np.linalg.inv(from_tp))
        _tr = np.eye(4)

        _tr[:3, :3] = Convert.Rotation.Rv_2_M(
            Convert.Rotation.M_2_Rv(_tp[:, :3, :3]).mean(axis=0))
        _tr[:3, 3] = _tp[:, :3, 3].mean(axis=0)

        return _tr

    # apply_transpose
    @staticmethod
    def Apply_extrinsic(
        pts: VEC_3 | VEC_4, extrinsic: TP, apply_inv: bool = False
    ) -> VEC_3:
        _ex = np.linalg.inv(extrinsic) if apply_inv else extrinsic
        _ex = _ex[:, :3, :]

        _ord = pts.shape[1]
        _pts: VEC_4 = Convert.To_homog(pts) if _ord == 3 else pts

        return np.einsum("bjk, nk -> bnk", _ex, _pts)

    @staticmethod
    def Remove_duplicate(
        data: TP, precision: int
    ):
        _scale: int = 10 ** precision
        _scaled = (data * _scale).round().astype(np.int64)  # 정수형 변환
        _, _indices = np.unique(_scaled, axis=0, return_index=True)

        return data[_indices], _indices
