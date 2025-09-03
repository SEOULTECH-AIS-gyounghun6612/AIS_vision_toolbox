"""카메라 모델 및 투영 계산을 위한 함수 모음."""

import numpy as np

from ..vision_types import VEC_2D, IMG_SIZE, IN_M, VEC_3D, IMG_1C

__all__ = [
    "Get_focal_length_from_fov", "Get_size_from_fov", "Get_fov_from_size",
    "Get_principal_point", "Get_principal_point_rate",
    "Compose_intrinsic_matrix", "Extract_intrinsic_params",
    "Adjust_intrinsic_matrix", "Apply_intrinsic_transform",
    "Get_points_from_depth", "Get_depth_map_from_points", "Remap_depth_map"
]


def Get_focal_length_from_fov(fov: VEC_2D, sz: IMG_SIZE) -> VEC_2D:
    """시야각(FOV)과 이미지 크기로부터 초점 거리를 계산합니다."""
    return sz / (2 * np.tan(np.deg2rad(fov) / 2))


def Get_size_from_fov(fov: VEC_2D, f_len: VEC_2D) -> IMG_SIZE:
    """시야각(FOV)과 초점 거리로부터 이미지 크기를 계산합니다."""
    _rads = np.deg2rad(fov) / 2
    _size = 2 * np.round(np.tan(_rads) * f_len)
    return _size.astype(int)


def Get_fov_from_size(sz: IMG_SIZE, f_len: VEC_2D) -> VEC_2D:
    """이미지 크기와 초점 거리로부터 시야각(FOV)을 계산합니다."""
    return 2 * np.rad2deg(np.arctan2(sz / 2, f_len))


def Get_principal_point(rate: VEC_2D, sz: IMG_SIZE) -> VEC_2D:
    """비율과 이미지 크기로부터 주점을 계산합니다."""
    return sz * rate


def Get_principal_point_rate(pp: VEC_2D, sz: IMG_SIZE) -> VEC_2D:
    """주점과 이미지 크기로부터 주점의 비율을 계산합니다."""
    return pp / sz


def Compose_intrinsic_matrix(f_len: VEC_2D, pp: VEC_2D) -> IN_M:
    """초점 거리와 주점으로부터 내부 파라미터 행렬을 구성합니다."""
    if f_len.ndim != pp.ndim or f_len.shape[0] != pp.shape[0]:
        raise ValueError("Dimensions of f_len and pp do not match.")
    
    _concat = np.concatenate([f_len, pp], axis=-1)
    if pp.ndim >= 2:
        _in_m = np.tile(np.eye(3), [pp.shape[0], 1, 1])
        _in_m[:, [0, 1, 0, 1], [0, 1, 2, 2]] = _concat
    else:
        _in_m = np.eye(3)
        _in_m[[0, 1, 0, 1], [0, 1, 2, 2]] = _concat
    return _in_m


def Extract_intrinsic_params(in_m: IN_M) -> tuple[VEC_2D, VEC_2D]:
    """내부 파라미터 행렬에서 초점 거리와 주점을 추출합니다."""
    if in_m.ndim >= 3:
        _p = in_m[:, [0, 1, 0, 1], [0, 1, 2, 2]]
        return _p[:, :2], _p[:, 2:]
    _p = in_m[[0, 1, 0, 1], [0, 1, 2, 2]]
    return _p[:2], _p[2:]


def Adjust_intrinsic_matrix(in_m: IN_M, sz: IMG_SIZE, new_sz: IMG_SIZE) -> IN_M:
    """이미지 크기 변경에 따라 내부 파라미터 행렬을 조정합니다."""
    _f_len, _pp = Extract_intrinsic_params(in_m)
    _fov = Get_fov_from_size(sz, _f_len)
    _pp_rate = Get_principal_point_rate(_pp, sz)
    _new_f_len = Get_focal_length_from_fov(_fov, new_sz)
    _new_pp = Get_principal_point(_pp_rate, new_sz)
    return Compose_intrinsic_matrix(_new_f_len, _new_pp)


def Apply_intrinsic_transform(
    pts: VEC_3D, in_m: IN_M, inv: bool = False
) -> VEC_3D:
    """포인트에 내부 파라미터 변환(투영)을 적용합니다."""
    _mat = np.linalg.inv(in_m) if inv else in_m
    return (_mat @ pts.T).T


def Get_points_from_depth(d_map: IMG_1C, mask: IMG_1C | None = None) -> VEC_3D:
    """깊이 맵으로부터 3D 포인트 클라우드를 생성합니다."""
    _h, _w = d_map.shape[:2]
    if mask is not None:
        _m = mask.squeeze()
    else:
        _m = np.ones((_h, _w), dtype=bool)
    
    _d = d_map[_m].squeeze()
    _yy, _xx = np.meshgrid(np.arange(_h), np.arange(_w), indexing='ij')
    return np.c_[_xx[_m] * _d, _yy[_m] * _d, _d]


def Get_depth_map_from_points(pts: VEC_3D, d_sz: tuple[int, int]) -> IMG_1C:
    """3D 포인트 클라우드로부터 깊이 맵을 생성합니다."""
    _w, _h = d_sz
    _vis_pts = pts[pts[:, 2] > 1e-6]
    _proj_pts = _vis_pts.copy()
    _proj_pts[:, :2] /= _proj_pts[:, 2][:, None]
    
    _u = np.round(_proj_pts[:, 0]).astype(np.int32)
    _v = np.round(_proj_pts[:, 1]).astype(np.int32)
    
    _m = (_u >= 0) & (_u < _w) & (_v >= 0) & (_v < _h)
    _u, _v, _d = _u[_m], _v[_m], _vis_pts[_m, 2]
    
    _d_flat = np.full(_h * _w, np.inf, dtype=np.float32)
    _sort_idx = np.argsort(-_d)
    _v_w_u = _v * _w + _u
    
    np.minimum.at(_d_flat, _v_w_u[_sort_idx], _d[_sort_idx])
    _d_flat[np.isinf(_d_flat)] = 0
    return _d_flat.reshape(_h, _w, 1)


def Remap_depth_map(depth: IMG_1C, in_m: IN_M, new_sz: tuple[int, int]) -> IMG_1C:
    """새로운 카메라 내부 파라미터에 맞게 깊이 맵을 리매핑합니다."""
    _h, _w = depth.shape[:2]
    _new_in = Adjust_intrinsic_matrix(
        in_m, np.array([_w, _h]), np.array(new_sz)
    )
    _pts_on_img = Get_points_from_depth(depth)
    _pts_on_cam = Apply_intrinsic_transform(_pts_on_img, in_m, inv=True)
    _new_pts_on_img = Apply_intrinsic_transform(_pts_on_cam, _new_in)
    return Get_depth_map_from_points(_new_pts_on_img, new_sz)