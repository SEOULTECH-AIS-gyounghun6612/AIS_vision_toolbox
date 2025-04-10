from enum import auto

from numpy import float32, ndarray, pad

import cv2

from python_ex.system import String


class Process():
    class Crop_Mode(String.String_Enum):
        PAD = auto()
        CENTER = auto()
        NEAR = auto()
        FAR = auto()

    @staticmethod
    def Adjust_width(
        img: ndarray, mode: Crop_Mode, gap: int, unit: int = 14
    ) -> ndarray:
        if mode == Process.Crop_Mode.NEAR:
            return img[:, :-gap]
        if mode == Process.Crop_Mode.FAR:
            return img[:, gap:]

        _st = gap // 2
        if mode == Process.Crop_Mode.CENTER:
            _ed = gap - _st
            return img[:, _st:-_ed]

        _ed = unit - _st
        return pad(img, ((0, 0), (_st, _ed)), constant_values=1.0)

    @staticmethod
    def Adjust_height(
        img: ndarray, mode: Crop_Mode, gap: int, unit: int = 14
    ) -> ndarray:
        if mode == Process.Crop_Mode.NEAR:
            return img[:-gap, :]
        if mode == Process.Crop_Mode.FAR:
            return img[gap:, :]

        _st = gap // 2

        if mode == Process.Crop_Mode.CENTER:
            _ed = gap - _st
            return img[_st:-_ed, :]

        _ed = unit - _st
        return pad(img, ((_st, _ed), (0, 0)), constant_values=1.0)


def Resize_img(
    file_name: str,
    is_width: bool = True,
    target_size: int = 518,
    unit: int = 14,
    scale: float = 255.
) -> tuple[ndarray, int]:
    _img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

    if _img.ndim != 2:
        # change to RGB
        _img = cv2.cvtColor(
            _img, cv2.COLOR_BGR2RGB if _img.ndim == 3 else cv2.COLOR_BGRA2RGB
        )

    _h, _w = _img.shape[:2]

    if is_width:
        _new_w = target_size
        _new_h = round(_h * (_new_w / _w))
        _gap = _new_h % unit
    else:
        _new_h = target_size
        _new_w = round(_w * (_new_h / _h))
        _gap = _new_w % unit

    return cv2.resize(_img, (_new_w, _new_h)).astype(float32) / scale, _gap
