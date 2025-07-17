


from __future__ import annotations
from typing import Type, ClassVar
from dataclasses import dataclass, field
from pathlib import Path

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

import cv2

from .utils.vision_types import (
    IN_M, TF_M, IMG_SIZE, VEC_1D, VEC_3D, VEC_4D, IMG_3C)

__all__ = ["Asset", "Camera", "Image", "Point_Cloud", "Scene", "Capture"]


@dataclass
class Asset(ABC):
    """저장/로드가 가능한 모든 데이터 객체의 추상 기본 클래스."""
    @abstractmethod
    def To_dict(self) -> dict:
        """객체를 직렬화 가능한 dict로 변환."""

    @classmethod
    @abstractmethod
    def From_dict(cls, data: dict) -> Asset:
        """dict로부터 객체를 생성."""

    @abstractmethod
    def Load_data(self, data_path: Path):
        """대용량 데이터를 메모리로 불러옴."""

    @abstractmethod
    def Save_data(self, data_path: Path):
        """대용량 데이터를 파일로 저장함."""


@dataclass
class Camera(Asset):
    """카메라의 고유한 내부 파라미터만 담는 Asset."""
    intrinsics: IN_M
    image_size: IMG_SIZE

    def To_dict(self) -> dict:
        return {
            "intrinsics": self.intrinsics.tolist(),
            "image_size": self.image_size.tolist()
        }

    @classmethod
    def From_dict(cls, data: dict) -> Camera:
        return cls(
            intrinsics=np.array(data["intrinsics"]),
            image_size=np.array(data["image_size"])
        )

    def Load_data(self, data_path: Path):
        pass  # No large data to load

    def Save_data(self, data_path: Path):
        pass  # No large data to save


@dataclass
class Image(Asset):
    """이미지 데이터와 파일 경로를 관리하는 Asset."""
    relative_path: str  # 데이터 소스 폴더 기준 상대 경로
    # 실제 이미지 데이터는 필요시 load_data()로 불러옴 (Lazy Loading)
    data: IMG_3C | None = field(default=None, repr=False)

    def To_dict(self) -> dict:
        return {"relative_path": self.relative_path}

    @classmethod
    def From_dict(cls, data: dict) -> Image:
        return cls(relative_path=data["relative_path"])

    def Load_data(self, data_path: Path):
        """지정된 소스 경로에서 이미지 데이터를 불러옴."""
        self.data = cv2.imread(str(data_path / self.relative_path))
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)

    def Save_data(self, data_path: Path):
        """지정된 소스 경로에 이미지 데이터를 저장."""
        if self.data is not None:
            _f_name = data_path / self.relative_path
            _f_name.parent.mkdir(exist_ok=True, parents=True)
            # OpenCV는 BGR 순서이므로 변환 후 저장
            cv2.imwrite(str(_f_name), cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR))


@dataclass
class Point_Cloud(Asset):
    """포인트 클라우드 데이터를 관리하는 Asset."""
    relative_path: str # 데이터 소스 폴더 기준 상대 경로 (npz)
    points: VEC_3D = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32), 
        repr=False
    )
    colors: VEC_3D = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32), 
        repr=False
    )

    def To_dict(self) -> dict:
        return {"relative_path": self.relative_path}

    @classmethod
    def From_dict(cls, data: dict) -> Point_Cloud:
        return cls(relative_path=data["relative_path"])

    def Load_data(self, data_path: Path):
        """npz 파일에서 포인트 클라우드 데이터를 불러옴."""
        with np.load(data_path / self.relative_path) as data:
            self.points = data['points']
            self.colors = data['colors']

    def Save_data(self, data_path: Path):
        """포인트 클라우드 데이터를 npz 파일로 저장."""
        if self.points is not None and self.colors is not None:
            _f_name = data_path / self.relative_path
            _f_name.parent.mkdir(exist_ok=True, parents=True)
            np.savez(_f_name, points=self.points, colors=self.colors)


@dataclass
class Gaussian_3DGS(Asset):
    """3D Gaussian Splatting 모델의 파라미터를 관리하는 Asset."""
    relative_path: str # 데이터 소스 폴더 기준 상대 경로 (npz)
    # --- mutable default를 안전하게 생성하기 위해 default_factory 사용 ---
    points: VEC_3D = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32),
        repr=False
    )
    opacities: VEC_1D = field(
        default_factory=lambda: np.empty((0, 1), dtype=np.float32),
        repr=False
    )
    scales: VEC_3D = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32),
        repr=False
    )
    rotations: VEC_4D = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32),
        repr=False
    )
    # 3DGS의 SH(Spherical Harmonics) 계수 또는 일반 RGB 값으로 활용 가능.
    # SH features는 (N, sh_degree**2, 3) 이지만, 편의상 2D로 저장/로드
    # 예: SH degree 3 (16) * 3 channels = 48
    color_feature: NDArray = field(
        default_factory=lambda: np.empty((0, 48), dtype=np.float32),
        repr=False
    )

    def To_dict(self) -> dict:
        return {"relative_path": self.relative_path}

    @classmethod
    def From_dict(cls, data: dict) -> Gaussian_3DGS:
        return cls(relative_path=data["relative_path"])

    def Load_data(self, data_path: Path):
        """npz 파일에서 3DGS 파라미터를 불러옴."""
        with np.load(data_path / self.relative_path) as data:
            self.points = data['points']
            self.opacities = data['opacities']
            self.scales = data['scales']
            self.rotations = data['rotations']
            self.color_feature = data['color_feature']

    def Save_data(self, data_path: Path):
        """3DGS 파라미터를 npz 파일로 저장."""
        if self.points is not None: # 모든 데이터 존재 여부 확인 필요
            _f_name = data_path / self.relative_path
            _f_name.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                _f_name,
                points=self.points,
                opacities=self.opacities,
                scales=self.scales,
                rotations=self.rotations,
                color_feature=self.color_feature
            )


@dataclass
class Scene(Asset):
    """
    하나의 시점(Pose)과 관측된 데이터 Asset들의 컨테이너.
    자신에게 속한 Asset들의 데이터 처리를 위임함.
    """
    ASSET_TYPE_REGISTRY: ClassVar[dict[str, Type[Asset]]] = {
        "Image": Image,
        "Point_Cloud": Point_Cloud,
        "Gaussian_3DGS": Gaussian_3DGS,
    }

    name: str
    pose: TF_M
    assets: dict[str, Asset] = field(default_factory=dict)

    def Load_data(self, data_path: Path) -> None:
        """자신이 가진 모든 Asset의 대용량 데이터를 불러옵니다."""
        print(f"\n--- Loading assets for Scene: {self.name} ---")
        for key, asset in self.assets.items():
            print(f"Loading asset '{key}'...")
            asset.Load_data(data_path)

    def Save_data(self, data_path: Path) -> None:
        """자신이 가진 모든 Asset의 대용량 데이터를 저장합니다."""
        print(f"\n--- Saving assets for Scene: {self.name} ---")
        for key, asset in self.assets.items():
            print(f"Saving asset '{key}'...")
            asset.Save_data(data_path)

    def To_dict(self) -> dict:
        """Scene 객체를 직렬화 가능한 dict로 변환합니다."""
        _data = {
            _k: {
                "asset_type": type(_a).__name__, 
                "data": _a.To_dict()
            } for _k, _a in self.assets.items()
        }
        return {"name": self.name, "pose": self.pose.tolist(), "assets": _data}

    @classmethod
    def From_dict(cls, data: dict) -> Scene:
        """dict로부터 Scene 객체를 생성합니다."""
        _assets, _reg = {}, cls.ASSET_TYPE_REGISTRY

        for _k, _info in data["assets"].items():
            _type_name = _info["asset_type"]
            _data = _info["data"]

            if _type_name not in _reg:
                print(f"Warning: Asset type missing for key '{_k}'. Skipping.")
                continue

            _assets[_k] = _reg[_type_name].From_dict(_data)

        return cls(
            name=data["name"],
            pose=np.array(data["pose"]),
            assets=_assets
        )

@dataclass
class Capture(Asset):
    """
    하나의 카메라와 배경 Scene을 기준으로, 여러 전경 Scene들을
    조합하여 최종 결과물들을 정의하는 최상위 객체.
    """
    name: str
    camera: Camera
    bg: Scene | None = None
    fg: dict[int, Scene] = field(default_factory=dict)

    # def get_effective_scenes(self) -> dict[int, Scene]:
    #     """
    #     배경과 모든 전경 Scene들을 조합하여
    #     최종 Scene들의 딕셔너리를 계산.
    #     """
    #     _scenes = {}

    #     _b = self.background_scene
    #     _b_pose, _b_asset = (_b.pose, _b.assets) if _b else (np.eye(4), {})

    #     for _fg_id, _fg_scene in self.foreground.items():
    #         _f_pose = _b_pose @ _fg_scene.pose
    #         _f_assets = {**_b_asset, **_fg_scene.assets}

    #         # 반환되는 Scene 객체는 이제 process_load/save 메서드를 가짐
    #         _scenes[_fg_id] = Scene(
    #             name=f"{self.name}_effective_{_fg_id}",
    #             pose=_f_pose,
    #             assets=_f_assets
    #         )
    #     return _scenes

    def Load_data(self, data_path: Path):
        """자신과 모든 하위 Scene들의 데이터를 불러옵니다."""
        print(f"--- Loading assets for Capture: {self.name} ---")
        if self.bg:
            self.bg.Load_data(data_path)
        for fg_scene in self.fg.values():
            fg_scene.Load_data(data_path)

    def Save_data(self, data_path: Path):
        """자신과 모든 하위 Scene들의 데이터를 저장합니다."""
        print(f"--- Saving assets for Capture: {self.name} ---")
        if self.bg:
            self.bg.Save_data(data_path)
        for fg_scene in self.fg.values():
            fg_scene.Save_data(data_path)


    def To_dict(self) -> dict:
        """Capture 객체를 직렬화 가능한 dict로 변환."""
        _b = self.bg
        _f = self.fg

        return {
            "name": self.name,
            "camera": self.camera.To_dict(),
            "bg": _b.To_dict() if _b else None,
            "fg": {
                fg_id: fg_scene.To_dict() for fg_id, fg_scene in _f.items()
            },
        }

    @classmethod
    def From_dict(cls, data: dict) -> Capture:
        """dict로부터 Capture 객체를 생성."""

        return cls(
            name=data["name"],
            camera=Camera.From_dict(data["camera"]),
            bg=Scene.From_dict(data["bg"]) if "bg" in data else None,
            fg={
                int(fg_id): Scene.From_dict(fg_data)
                for fg_id, fg_data in data.get("fg", {}).items()
            },
        )