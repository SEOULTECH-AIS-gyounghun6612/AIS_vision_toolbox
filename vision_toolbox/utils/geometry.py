"""범용 기하학 연산 수행 기능 포함."""
from typing import Literal
from scipy.spatial.transform import Rotation as R
import numpy as np
from .vision_types import ROT_M, VEC_4D, VEC_3D, VEC_2D, TF_M

__all__ = ["Convert", "L_TO_R"]

L_TO_R = np.array([ # 좌표계 변환(Left-to-Right Handed) 상수
    [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    [-0.0, -0.0, -1.0, -0.0], [0.0, 0.0, 0.0, 1.0]
])

class Convert:
    """회전, 좌표계 변환 등 기하학 데이터 변환 클래스."""
    @staticmethod
    def From_matrix_to_quat(mat: ROT_M) -> VEC_4D:
        """회전 행렬 -> 쿼터니언."""
        return R.from_matrix(mat).as_quat()

    @staticmethod
    def From_quat_to_matrix(quat: VEC_4D) -> ROT_M:
        """쿼터니언 -> 회전 행렬."""
        return R.from_quat(quat).as_matrix()

    @staticmethod
    def From_matrix_to_rotvec(mat: ROT_M) -> VEC_3D:
        """회전 행렬 -> 회전 벡터."""
        return R.from_matrix(mat).as_rotvec(degrees=True)

    @staticmethod
    def From_rotvec_to_matrix(r_vec: VEC_3D) -> ROT_M:
        """회전 벡터 -> 회전 행렬."""
        return R.from_rotvec(r_vec, degrees=True).as_matrix()

    @staticmethod
    def To_homogeneous(pts: VEC_2D | VEC_3D) -> VEC_3D | VEC_4D:
        """포인트 -> 동차 좌표."""
        return np.c_[pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)]

    @staticmethod
    def Change_handedness(
        obj: VEC_3D | VEC_4D | TF_M, mode: Literal["L2R", "R2L"] = "L2R"
    ):
        """좌표계(Handedness) 변환."""
        if mode != "L2R":
            raise NotImplementedError("R2L 모드는 아직 구현되지 않음.")
        tf_mat = L_TO_R

        if obj.ndim == 2: # 포인트(VEC_3D, VEC_4D)인 경우
            pts_h = Convert.To_homogeneous(obj) if obj.shape[1] == 3 else obj
            return (tf_mat @ pts_h.T).T[:, :3]

        return tf_mat @ obj # 변환 행렬(TF_M)인 경우
