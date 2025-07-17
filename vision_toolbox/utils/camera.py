"""카메라 모델, 파라미터, 투영/역투영 관련 기능 포함."""
import numpy as np
from .vision_types import VEC_2D, IMG_SIZE, IN_M, VEC_3D, IMG_1C

__all__ = ["Camera"]

class Camera:
    """카메라 모델 관련 계산 클래스."""
    @staticmethod
    def Get_focal_length_from(fov: VEC_2D, sz: IMG_SIZE) -> VEC_2D:
        """화각, 이미지 크기 -> 촛점 거리."""
        return sz / (2 * np.tan(np.deg2rad(fov) / 2))

    @staticmethod
    def Get_size_from(fov: VEC_2D, f_len: VEC_2D) -> IMG_SIZE:
        """화각, 촛점 거리 -> 이미지 크기."""
        return (2 * np.round(np.tan(np.deg2rad(fov)/2) * f_len)).astype(int)

    @staticmethod
    def Get_fov_from(sz: IMG_SIZE, f_len: VEC_2D) -> VEC_2D:
        """이미지 크기, 촛점 거리 -> 화각."""
        return 2 * np.rad2deg(np.arctan2(sz / 2, f_len))

    @staticmethod
    def Get_pp_from(rate: VEC_2D, sz: IMG_SIZE) -> VEC_2D:
        """비율, 이미지 크기 -> 주점(Principal Point)."""
        return sz * rate

    @staticmethod
    def Get_pp_rate_from(pp: VEC_2D, sz: IMG_SIZE) -> VEC_2D:
        """주점, 이미지 크기 -> 주점 비율."""
        return pp / sz

    @staticmethod
    def Compose_intrinsic(f_len: VEC_2D, pp: VEC_2D) -> IN_M:
        """촛점거리, 주점 -> 내부 파라미터 행렬."""
        if f_len.ndim != pp.ndim or f_len.shape[0] != pp.shape[0]:
            raise ValueError("f_len과 pp의 차원 또는 개수가 일치하지 않음.")
        concat = np.concatenate([f_len, pp], axis=-1)
        if pp.ndim >= 2:
            in_m = np.tile(np.eye(3), [pp.shape[0], 1, 1])
            in_m[:, [0, 1, 0, 1], [0, 1, 2, 2]] = concat
        else:
            in_m = np.eye(3)
            in_m[[0, 1, 0, 1], [0, 1, 2, 2]] = concat
        return in_m

    @staticmethod
    def Extract_intrinsic(in_m: IN_M) -> tuple[VEC_2D, VEC_2D]:
        """내부 파라미터 행렬 -> 촛점거리, 주점."""
        if in_m.ndim >= 3:
            p = in_m[:, [0, 1, 0, 1], [0, 1, 2, 2]]
            return p[:, :2], p[:, 2:]
        p = in_m[[0, 1, 0, 1], [0, 1, 2, 2]]
        return p[:2], p[2:]

    @staticmethod
    def Adjust_intrinsic(
        in_m: IN_M, sz: IMG_SIZE, new_sz: IMG_SIZE
    ) -> IN_M:
        """이미지 리사이즈에 맞춰 내부 파라미터 행렬 조정."""
        f_len, pp = Camera.Extract_intrinsic(in_m)
        fov = Camera.Get_fov_from(sz, f_len)
        pp_rate = Camera.Get_pp_rate_from(pp, sz)
        new_f_len = Camera.Get_focal_length_from(fov, new_sz)
        new_pp = Camera.Get_pp_from(pp_rate, new_sz)
        return Camera.Compose_intrinsic(new_f_len, new_pp)

    @staticmethod
    def Apply_intrinsic(
        pts: VEC_3D, in_m: IN_M, inv: bool = False
    ) -> VEC_3D:
        """포인트에 내부 파라미터(또는 역행렬) 적용."""
        mat = np.linalg.inv(in_m) if inv else in_m
        return (mat @ pts.T).T

    @staticmethod
    def Get_pts_from(d_map: IMG_1C, mask: IMG_1C | None = None) -> VEC_3D:
        """뎁스맵 -> 3D 포인트 (이미지 평면 좌표계)."""
        h, w = d_map.shape[:2]
        m = mask if mask is not None else np.ones((h,w), dtype=np.bool_)
        d = d_map[m, 0]
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        return np.c_[xx[m] * d, yy[m] * d, d]

    @staticmethod
    def Get_depth_map_from(pts: VEC_3D, d_sz: tuple[int, int]) -> IMG_1C:
        """3D 포인트 -> 뎁스맵."""
        w, h = d_sz
        vis_pts = pts[pts[:, 2] > 0]
        vis_pts[:, :2] /= vis_pts[:, 2][:, None]

        u = np.round(vis_pts[:, 0]).astype(np.int32)
        v = np.round(vis_pts[:, 1]).astype(np.int32)
        m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, d = u[m], v[m], vis_pts[m, 2]

        d_flat = np.full(h * w, np.inf, dtype=np.float32)
        np.minimum.at(d_flat, (v * w + u), d)
        d_flat = np.nan_to_num(d_flat, nan=0., posinf=0., neginf=0.)
        d_flat[d_flat < 0] = 0.0
        return d_flat.reshape(h, w)[:, :, None]

    @staticmethod
    def Remapping_depth_map(
        depth: IMG_1C, in_m: IN_M, new_sz: tuple[int, int]
    ):
        """뎁스맵을 새로운 내부 파라미터에 맞게 리매핑."""
        h, w = depth.shape[:2]
        new_in = Camera.Adjust_intrinsic(in_m, (w, h), new_sz)
        pts_on_img = Camera.Get_pts_from(depth)
        pts_on_cam = Camera.Apply_intrinsic(pts_on_img, in_m, True)
        new_pts_on_img = Camera.Apply_intrinsic(pts_on_cam, new_in)
        return Camera.Get_depth_map_from(new_pts_on_img, new_sz)
