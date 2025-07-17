"""이미지 처리 관련 기능 포함."""
from typing import Literal
import numpy as np
import cv2

from .vision_types import IMG_1C, IMG_3C

__all__ = ["Image"]

CROP_MODE = Literal["pad", "crop_n", "crop_f", "crop_c"]


class Image:
    """이미지 처리 클래스."""
    @staticmethod
    def Get_new_shape(
        sz: tuple[int, int], ref_sz: int, unit: int = 14,
        by_w: bool = True, use_pad: bool = True
    ) -> tuple[tuple[int, int], int]:
        """리사이즈 목표 크기와 갭 계산."""
        ref = sz[int(not by_w)]
        rate = ref_sz / ref
        target = round(sz[int(by_w)] * rate)
        gap = unit - (target % unit) if use_pad else - (target % unit)
        return ((ref_sz, target), gap) if by_w else ((target, ref_sz), gap)

    @staticmethod
    def Adjust_size(
        img: IMG_1C | IMG_3C, mode: CROP_MODE, gap: int,
        is_w: bool = True, fill: int | float = 0
    ) -> np.ndarray:
        """계산된 갭 만큼 이미지를 패딩 또는 크롭."""
        if mode == "crop_n":
            return img[:, :gap] if is_w else img[:gap, :]
        if mode == "crop_f":
            return img[:, -gap:] if is_w else img[-gap:, :]

        _gap = -gap if gap < 0 else gap
        st, ed = _gap // 2, _gap - (_gap // 2)
        if mode == "crop_c":
            return img[:, st:-ed] if is_w else img[st:-ed, :]

        pad_v = ((0, 0), (st, ed), (0, 0)) if is_w else \
                ((st, ed), (0, 0), (0, 0))
        return np.pad(img, pad_v, constant_values=fill)

    @staticmethod
    def Resize_img_with_gap(
        img: IMG_1C | IMG_3C, sz: tuple[int, int], mode: CROP_MODE, gap: int,
        is_w: bool = False
    ) -> IMG_1C | IMG_3C:
        """이미지 리사이즈 후 갭 처리."""
        r_img = cv2.resize(img, sz)
        if gap:
            return Image.Adjust_size(r_img, mode, gap, is_w, 255)
        return r_img

    @staticmethod
    def Resize_img(
        img: IMG_1C | IMG_3C, mode: CROP_MODE,
        ref: int = 518, unit: int = 14, by_w: bool = True
    ) -> IMG_1C | IMG_3C:
        """조건에 맞춰 이미지 리사이즈."""
        sz, pad_gap = Image.Get_new_shape(
            img.shape[-2::-1], ref, unit, by_w, mode == "pad")
        return Image.Resize_img_with_gap(img, sz, mode, pad_gap, not by_w)

    @staticmethod
    def Visualize_image_with_threshold(
        img: IMG_1C, threshold: tuple[float, float] | None = None,
        cmap=cv2.COLORMAP_JET, vis_gnd: bool = False
    ) -> np.ndarray:
        """임계값 기준으로 1채널 이미지 시각화."""
        if threshold:
            h, l = max(threshold), min(threshold)
            img[(img < l) & (img > h)] = 0.0
        _min, _max = np.min(img), np.max(img)
        if _min == _max:
            _img = (np.ones_like(img) * 255).astype(np.uint8)
        else:
            _img = np.round(
                (img - _min) / (_max - _min) * 255).astype(np.uint8)
        
        colored_d = cv2.applyColorMap(_img, cmap)
        if not vis_gnd:
            colored_d[_img[..., 0] == 0] = [0, 0, 0]
        return colored_d
