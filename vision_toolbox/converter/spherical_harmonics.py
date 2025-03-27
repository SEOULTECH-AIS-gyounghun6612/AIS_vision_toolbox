from numpy import (ndarray, uint8, ones, c_, newaxis)
from numpy.linalg import norm

# SH (Spherical Harmonics) 계수 정의
SH0 = 0.282095
SH1 = 0.488603
SH2 = (1.092548, 0.315392, 0.546274)
SH3 = (0.590044, 2.890611, 0.457046, 0.373176, 0.457046, 1.445306)
SH4 = (2.503343, 1.770131, 0.946175, 0.669047, 0.105785, 0.473087)


def Get_sh_bias_from(norm_pts: ndarray, max_l: int) -> ndarray:
    """
    ### 함수 개요
    - 단위 벡터(norm_pts)를 입력받아 구면조화계수(Spherical Harmonics)를 기준으로 bias 벡터를 생성합니다.
    - l 차수에 따라 SH 계수를 확장적으로 적용합니다. (l=0 ~ l=4 지원)

    ---
    ### Args
    - `max_l` (int): 확장할 SH 계수의 최대 차수 (0~4 사이 권장)
    - `norm_pts` (ndarray): (N, 3) 크기의 정규화된 3D 좌표 배열

    ### Returns
    - `ndarray`: (N, D) 크기의 SH bias 행렬. (D = (max_l + 1)^2).

    ### Raises
    - `ValueError`: 입력 차수가 0 미만이거나 norm_pts의 shape가 올바르지 않은 경우
    """
    if max_l < 0 or norm_pts.ndim != 2 or norm_pts.shape[1] != 3:
        raise ValueError("max_l은 0 이상이어야 하며 norm_pts는 (N, 3) 형태여야 합니다.")

    _bias = [ones(norm_pts.shape[0]) * SH0]

    if max_l:
        # max_l = 1
        _x, _y, _z = norm_pts[:, 0], norm_pts[:, 1], norm_pts[:, 2]
        _bias.extend([SH1 * _y, SH1 * _z, SH1 * _x])

        if max_l > 1:
            x2, y2, z2 = _x**2, _y**2, _z**2
            xy, yz, zx = _x * _y, _y * _z, _z * _x
            z2t1 = 3 * z2 - 1
            x2y2t1 = x2 - y2
            _bias.extend([
                SH2[0] * xy,
                SH2[0] * yz,
                SH2[1] * z2t1,
                SH2[0] * zx,
                SH2[2] * x2y2t1
            ])

            if max_l > 2:
                z2t2 = 5 * z2 - 1
                z2t3 = z2t2 - 2
                x2y2t2 = 3 * x2 - y2
                x2y2t3 = x2 - 3 * y2
                _bias.extend([
                    SH3[0] * _y * x2y2t2,
                    SH3[1] * xy * _z,
                    SH3[2] * _y * z2t2,
                    SH3[3] * _z * z2t3,
                    SH3[4] * _x * z2t2,
                    SH3[5] * _z * x2y2t1,
                    SH3[0] * _x * x2y2t3
                ])

                if max_l > 3:
                    z2t4 = 7 * z2 - 1
                    z2t5 = _z * (7 * z2 - 3)
                    _bias.extend([
                        SH4[0] * xy * x2y2t1,
                        SH4[1] * yz * x2y2t2,
                        SH4[2] * xy * z2t4,
                        SH4[3] * _y * z2t5,
                        SH4[4] * (7 * z2 * z2t3 - 3 * z2t1),
                        SH4[3] * _x * z2t5,
                        SH4[5] * x2y2t1 * z2t4,
                        SH4[1] * xy * x2y2t3,
                        SH4[3] * (x2 * x2y2t3 - y2 * x2y2t2)
                    ])

    return c_[*_bias]


def Convert_to_rgb_from(pts: ndarray, sh_weight: ndarray, max_l: int):
    _len = max_l * max_l
    _bias = Get_sh_bias_from(
        pts / norm(pts, axis=1, keepdims=True), max_l)[:, :_len]

    _rgb = (sh_weight[:, :, :_len] @ _bias[:, :, newaxis])[:, :, 0]

    _min: ndarray = _rgb.min(axis=0)
    _max: ndarray = _rgb.max(axis=0)

    return ((_rgb - _min) / (_max - _min) * 255).astype(uint8)
