"""카메라 자세(Pose), 외부 파라미터(Extrinsic) 관련 기능 포함."""
import numpy as np
from .vision_types import TF_M, VEC_4D, VEC_3D
from .geometry import Convert

__all__ = ["Pose"]

class Pose:
    """카메라 자세 관련 계산 클래스."""
    @staticmethod
    def Compose_extrinsic(q: VEC_4D, t: VEC_3D) -> TF_M:
        """쿼터니언, 이동 벡터 -> 외부 파라미터 행렬."""
        mat = Convert.From_quat_to_matrix(q)
        if q.ndim >= 2:
            ext = np.tile(np.eye(4), [q.shape[0], 1, 1])
            ext[:, :3, :3], ext[:, :3, 3] = mat, t
        else:
            ext = np.eye(4)
            ext[:3, :3], ext[:3, 3] = mat, t
        return ext

    @staticmethod
    def Extract_extrinsic(ext: TF_M) -> tuple[VEC_4D, VEC_3D]:
        """외부 파라미터 행렬 -> 쿼터니언, 이동 벡터."""
        if ext.ndim >= 3:
            mat, t = ext[:, :3, :3], ext[:, :3, 3]
        else:
            mat, t = ext[:3, :3], ext[:3, 3]
        q = Convert.From_matrix_to_quat(mat)
        return q, t

    @staticmethod
    def Get_median_extrinsic(
        from_tfs: TF_M, to_tfs: TF_M
    ) -> TF_M:
        """두 자세 집합 간의 상대 변환들의 평균(중앙값) 계산."""
        assert len(from_tfs) == len(to_tfs)
        rel_tfs = to_tfs @ np.linalg.inv(from_tfs)
        tr = np.eye(4)
        
        r_vecs = Convert.From_matrix_to_rotvec(rel_tfs[:, :3, :3])
        tr[:3, :3] = Convert.From_rotvec_to_matrix(r_vecs.mean(axis=0))
        tr[:3, 3] = rel_tfs[:, :3, 3].mean(axis=0)
        return tr
    
    @staticmethod
    def Apply_extrinsic(
        pts: VEC_3D | VEC_4D, ext: TF_M, inv: bool = False
    ) -> VEC_3D:
        """포인트에 외부 파라미터(자세) 적용."""
        _ext = np.linalg.inv(ext) if inv else ext
        _ext = _ext[:, :3, :]
        pts_h = Convert.To_homogeneous(pts) if pts.shape[1] == 3 else pts
        return np.einsum("bjk,nk->bnk", _ext, pts_h)

    @staticmethod
    def Remove_duplicate(
        data: TF_M, precision: int
    ) -> tuple[TF_M, np.ndarray]:
        """지정된 정밀도에 따라 중복된 자세 데이터 제거."""
        scale: int = 10 ** precision
        scaled = (data * scale).round().astype(np.int64)
        _, indices = np.unique(scaled, axis=0, return_index=True)
        return data[indices], indices