"""범용 기하학 연산 및 변환을 위한 함수 모음."""

from typing import Literal
from scipy.spatial.transform import Rotation as R
import numpy as np

from ..vision_types import (
    ROT_M, VEC_4D, VEC_3D, VEC_2D, TF_M
)

__all__ = [
    "L_TO_R", "Compute_3d_covariance", "From_matrix_to_quat",
    "From_quat_to_matrix", "From_matrix_to_rotvec", "From_rotvec_to_matrix",
    "To_homogeneous", "Change_handedness", "Compose_extrinsic_matrix",
    "Extract_extrinsic_params", "Get_median_extrinsic", 
    "Apply_extrinsic_transform", "Remove_duplicate_poses"
]

L_TO_R = np.array([ # 좌표계 변환(Left-to-Right Handed) 상수
    [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    [-0.0, -0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]
])


def Compute_3d_covariance(scales: VEC_3D, rot: VEC_4D) -> tuple[VEC_3D, VEC_3D]:
    """3D 가우시안의 공분산 행렬의 상삼각행렬 성분을 계산합니다."""
    _norm = np.linalg.norm(rot, axis=1, keepdims=True)
    _norm_rot = rot / _norm
    _mat = R.from_quat(_norm_rot).as_matrix()

    _s = np.zeros((scales.shape[0], 3, 3), dtype=np.float32)
    _s[:, 0, 0] = scales[:, 0]
    _s[:, 1, 1] = scales[:, 1]
    _s[:, 2, 2] = scales[:, 2]

    _m = _mat @ _s
    _sigma = _m @ np.transpose(_m, (0, 2, 1))

    _covA = np.stack([
        _sigma[:, 0, 0], _sigma[:, 0, 1], _sigma[:, 0, 2]
    ], axis=1)
    _covB = np.stack([
        _sigma[:, 1, 1], _sigma[:, 1, 2], _sigma[:, 2, 2]
    ], axis=1)

    return _covA, _covB


def From_matrix_to_quat(mat: ROT_M) -> VEC_4D:
    """회전 행렬 -> 쿼터니언."""
    return R.from_matrix(mat).as_quat()


def From_quat_to_matrix(quat: VEC_4D) -> ROT_M:
    """쿼터니언 -> 회전 행렬."""
    return R.from_quat(quat).as_matrix()


def From_matrix_to_rotvec(mat: ROT_M) -> VEC_3D:
    """회전 행렬 -> 회전 벡터."""
    return R.from_matrix(mat).as_rotvec(degrees=True)


def From_rotvec_to_matrix(r_vec: VEC_3D) -> ROT_M:
    """회전 벡터 -> 회전 행렬."""
    return R.from_rotvec(r_vec, degrees=True).as_matrix()


def To_homogeneous(pts: VEC_2D | VEC_3D) -> VEC_3D | VEC_4D:
    """포인트 -> 동차 좌표."""
    _ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    return np.c_[pts, _ones]


def Change_handedness(
    obj: VEC_3D | VEC_4D | TF_M, mode: Literal["L2R", "R2L"] = "L2R"
):
    """좌표계(Handedness) 변환."""
    if mode != "L2R":
        raise NotImplementedError("R2L 모드는 아직 구현되지 않음.")

    if obj.ndim == 2: # 포인트(VEC_3D, VEC_4D)인 경우
        _pts_h = To_homogeneous(obj) if obj.shape[1] == 3 else obj
        return (L_TO_R @ _pts_h.T).T[:, :3]

    return L_TO_R @ obj # 변환 행렬(TF_M)인 경우


def Compose_extrinsic_matrix(q: VEC_4D, t: VEC_3D) -> TF_M:
    """쿼터니언과 이동 벡터로 외부 파라미터 행렬을 구성합니다."""
    _mat = From_quat_to_matrix(q)
    if q.ndim >= 2:
        _ext = np.tile(np.eye(4), [q.shape[0], 1, 1])
        _ext[:, :3, :3], _ext[:, :3, 3] = _mat, t
    else:
        _ext = np.eye(4)
        _ext[:3, :3], _ext[:3, 3] = _mat, t
    return _ext


def Extract_extrinsic_params(ext: TF_M) -> tuple[VEC_4D, VEC_3D]:
    """외부 파라미터 행렬에서 쿼터니언과 이동 벡터를 추출합니다."""
    if ext.ndim >= 3:
        _mat, _t = ext[:, :3, :3], ext[:, :3, 3]
    else:
        _mat, _t = ext[:3, :3], ext[:3, 3]
    _q = From_matrix_to_quat(_mat)
    return _q, _t


def Get_median_extrinsic(from_tfs: TF_M, to_tfs: TF_M) -> TF_M:
    """변환 행렬들 사이의 중간값에 해당하는 변환 행렬을 계산합니다."""
    assert len(from_tfs) == len(to_tfs)
    _rel_tfs = to_tfs @ np.linalg.inv(from_tfs)
    _tr = np.eye(4)
    _r_vecs = R.from_matrix(_rel_tfs[:, :3, :3]).as_rotvec()
    _tr[:3, :3] = R.from_rotvec(_r_vecs.mean(axis=0)).as_matrix()
    _tr[:3, 3] = _rel_tfs[:, :3, 3].mean(axis=0)
    return _tr


def Apply_extrinsic_transform(
    pts: VEC_3D | VEC_4D, ext: TF_M, inv: bool = False
) -> VEC_3D:
    """포인트에 외부 파라미터 변환을 적용합니다."""
    _ext = np.linalg.inv(ext) if inv else ext
    _ext = _ext[..., :3, :]
    _pts_h = To_homogeneous(pts) if pts.shape[-1] == 3 else pts
    _transformed_pts = np.einsum("...ij,...j->...i", _ext, _pts_h)
    return _transformed_pts


def Remove_duplicate_poses(
    data: TF_M, precision: int
) -> tuple[TF_M, np.ndarray]:
    """중복되는 포즈들을 제거합니다."""
    _scale: int = 10 ** precision
    _flat_data = data.reshape(data.shape[0], -1)
    _scaled = (_flat_data * _scale).round().astype(np.int64)
    _, _indices = np.unique(_scaled, axis=0, return_index=True)
    return data[_indices], _indices