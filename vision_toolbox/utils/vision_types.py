"""프로젝트 공통 타입 별칭(상수) 정의."""

from typing import Annotated
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ROT_M", "TRANS_V", "TF_M", "IN_M", "VEC_2D", "VEC_3D", "VEC_4D",
    "IMG_SIZE", "PT_COLOR", "IMG_1C", "IMG_3C"
]

# 행렬 및 파라미터 타입
ROT_M = Annotated[NDArray, (None, 3, 3)]    # 회전 행렬
TRANS_V = Annotated[NDArray, (None, 3, 1)]  # 이동 벡터
TF_M = Annotated[NDArray, (None, 4, 4)]     # 변환 행렬 (자세)
IN_M = Annotated[NDArray, (None, 3, 3)]     # 카메라 내부 파라미터

# 벡터 타입
VEC_1D = Annotated[NDArray, (None, 1)]  # 1D 벡터
VEC_2D = Annotated[NDArray, (None, 2)]  # 2D 벡터
VEC_3D = Annotated[NDArray, (None, 3)]  # 3D 벡터
VEC_4D = Annotated[NDArray, (None, 4)]  # 4D 벡터 (동차 좌표)

# 이미지 및 포인트 타입
IMG_SIZE = Annotated[NDArray[np.int32], (None, 2)]  # 이미지 크기
PT_COLOR = Annotated[NDArray[np.uint8], (None, 3)]  # 포인트 색상
IMG_1C = Annotated[NDArray, (None, None, 1)]        # 1채널 이미지
IMG_3C = Annotated[NDArray, (None, None, 3)]        # 3채널 이미지
