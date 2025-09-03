"""이미지 처리 연산을 위한 함수 모음."""

from typing import Literal
import numpy as np
import cv2

from ..vision_types import IMG_3C, IMG_1C

__all__ = [
    "CROP_MODE", "Get_new_shape", "Adjust_size", "Resize_with_gap",
    "Resize_image", "Visualize_image"
]

CROP_MODE = Literal["pad", "crop_n", "crop_f", "crop_c"]


def Get_new_shape(
    sz: tuple[int, int], 
    ref_sz: int, 
    unit: int = 14, 
    by_w: bool = True, 
    use_pad: bool = True
) -> tuple[tuple[int, int], int]:
    """참조 크기, 단위(unit)에 맞춰 새로운 이미지 크기와 패딩/크롭 갭을 계산."""
    _w, _h = sz
    _ref = _w if by_w else _h
    _other_dim = _h if by_w else _w
    _rate = ref_sz / _ref
    _target_other_dim = round(_other_dim * _rate)
    _dim_to_check = _target_other_dim if by_w else ref_sz
    _gap = (unit - (_dim_to_check % unit)) % unit
    if not use_pad:
        _gap = - (_dim_to_check % unit) if _dim_to_check % unit != 0 else 0
    
    _new_sz_wh = (ref_sz, _target_other_dim) if by_w else (_target_other_dim, ref_sz)
    return _new_sz_wh, _gap


def Adjust_size(
    img: IMG_3C | IMG_1C, 
    mode: CROP_MODE, 
    gap: int, 
    is_w_dim: bool, 
    fill: int | float = 0
) -> np.ndarray:
    """계산된 갭(gap)만큼 이미지에 패딩을 추가하거나 크롭합니다."""
    if gap == 0: return img
    
    _abs_gap = abs(gap)
    if gap < 0:
        if mode == "crop_n": 
            return img[:, :-_abs_gap] if is_w_dim else img[:-_abs_gap, :]
        if mode == "crop_f": 
            return img[:, _abs_gap:] if is_w_dim else img[_abs_gap:, :]
        
        _st, _ed = _abs_gap // 2, _abs_gap - (_abs_gap // 2)
        return img[:, _st:-_ed] if is_w_dim else img[_st:-_ed, :]
    
    _st, _ed = gap // 2, gap - (gap // 2)
    _pad_v = ((0, 0), (_st, _ed), (0, 0)) if is_w_dim else ((_st, _ed), (0, 0), (0, 0))
    _pad_dims = _pad_v[:img.ndim]
    return np.pad(img, _pad_dims, 'constant', constant_values=fill)


def Resize_with_gap(
    img: IMG_3C | IMG_1C, 
    sz: tuple[int, int], 
    mode: CROP_MODE, 
    gap: int, 
    is_gap_on_w: bool
) -> np.ndarray:
    """이미지를 리사이즈하고 갭만큼 조정합니다."""
    _r_img = cv2.resize(img, sz, interpolation=cv2.INTER_AREA)
    if _r_img.ndim == 2: 
        _r_img = _r_img[..., None]
    return Adjust_size(_r_img, mode, gap, is_gap_on_w)


def Resize_image(
    img: IMG_3C | IMG_1C, 
    mode: CROP_MODE, 
    ref: int, 
    unit: int, 
    by_w: bool
) -> np.ndarray:
    """주어진 조건에 맞게 이미지를 최종 리사이즈합니다."""
    _h, _w = img.shape[:2]
    _sz_wh, _pad_gap = Get_new_shape((_w, _h), ref, unit, by_w, mode == "pad")
    return Resize_with_gap(img, _sz_wh, mode, _pad_gap, not by_w)


def Visualize_image(
    img: IMG_1C, 
    v_min: float | None = None, 
    v_max: float | None = None, 
    cmap=cv2.COLORMAP_JET, 
    invalid_color=None
) -> np.ndarray:
    """1채널 이미지를 컬러맵을 적용하여 시각화합니다."""
    if invalid_color is None: 
        invalid_color = [0, 0, 0]
    
    _valid_mask = (img > 0) & np.isfinite(img)
    _valid_pixels = img[_valid_mask]
    
    if v_min is None: 
        _v_min = np.min(_valid_pixels) if len(_valid_pixels) > 0 else 0
    if v_max is None: 
        _v_max = np.max(_valid_pixels) if len(_valid_pixels) > 0 else 1
    
    _range = max(_v_max - _v_min, 1e-8)
    _img_norm = np.clip((img - _v_min) / _range, 0, 1)
    _img_u8 = (_img_norm * 255).astype(np.uint8)
    
    _colored = cv2.applyColorMap(_img_u8.squeeze(), cmap)
    _colored[~_valid_mask.squeeze()] = invalid_color
    return _colored