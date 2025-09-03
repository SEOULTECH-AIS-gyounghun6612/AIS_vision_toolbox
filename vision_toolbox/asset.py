"""vision asset과 scene 구조 관리를 위한 데이터 모델.

이 클래스들은 순수한 데이터 컨테이너입니다. 모든 파일 I/O 및 처리 연산은
'vision.core' 모듈에서 처리합니다.
"""

from __future__ import annotations
from typing import Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

from .vision_types import IN_M, TF_M, IMG_SIZE, VEC_3D, IMG_3C

__all__ = [
    "Asset", "Image", "Point_Cloud", "Camera", "Scene"
]

# 타입 이름을 클래스에 매핑하는 레지스트리. 직렬화 해제(deserialization) 시 사용됩니다.
ASSET_CLASS_REGISTRY: dict[str, Type[Asset]] = {}

def _Register_asset_type(cls: Type[Asset]) -> Type[Asset]:
    """Asset 하위 클래스를 직렬화를 위해 자동으로 등록하는 데코레이터."""
    ASSET_CLASS_REGISTRY[cls.__name__] = cls
    return cls

@dataclass
class Asset(ABC):
    """모든 직렬화 가능한 데이터 객체의 추상 기본 클래스."""
    @abstractmethod
    def To_dict(self) -> dict:
        """객체를 딕셔너리로 직렬화합니다."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def From_dict(cls, data: dict) -> Asset:
        """딕셔너리로부터 객체를 생성합니다."""
        raise NotImplementedError

@_Register_asset_type
@dataclass
class Image(Asset):
    """이미지 데이터용 Asset. 'data' 필드는 I/O 함수에 의해 채워집니다."""
    relative_path: str
    data: IMG_3C | None = field(default=None, repr=False)

    def To_dict(self) -> dict:
        return {"relative_path": self.relative_path}

    @classmethod
    def From_dict(cls, data: dict) -> Image:
        return cls(relative_path=data["relative_path"])

@_Register_asset_type
@dataclass
class Point_Cloud(Asset):
    """포인트 클라우드 데이터용 Asset."""
    relative_path: str
    points: VEC_3D = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32), repr=False)
    colors: VEC_3D = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.uint8), repr=False)

    def To_dict(self) -> dict:
        return {"relative_path": self.relative_path}

    @classmethod
    def From_dict(cls, data: dict) -> Point_Cloud:
        return cls(relative_path=data["relative_path"])

@_Register_asset_type
@dataclass
class Camera(Asset):
    """내부 파라미터, 자세, 그리고 연관된 asset들을 포함하는 단일 카메라 컨테이너."""
    name: str
    pose: TF_M
    intrinsics: IN_M
    image_size: IMG_SIZE
    assets: dict[str, Asset] = field(default_factory=dict)

    def To_dict(self) -> dict:
        return {
            "name": self.name,
            "pose": self.pose.tolist(),
            "intrinsics": self.intrinsics.tolist(),
            "image_size": self.image_size.tolist(),
            "assets": {
                key: {
                    "asset_type": type(asset).__name__,
                    "data": asset.To_dict()
                }
                for key, asset in self.assets.items()
            }
        }

    @classmethod
    def From_dict(cls, data: dict) -> Camera:
        assets = {}
        for key, info in data.get("assets", {}).items():
            asset_class = ASSET_CLASS_REGISTRY.get(info["asset_type"])
            if asset_class:
                assets[key] = asset_class.From_dict(info["data"])

        return cls(
            name=data["name"],
            pose=np.array(data["pose"]),
            intrinsics=np.array(data["intrinsics"]),
            image_size=np.array(data["image_size"]),
            assets=assets
        )

@dataclass
class Scene(Asset):
    """3D asset과 카메라 컬렉션을 포함하는 프로젝트 최상위 컨테이너."""
    name: str
    assets: dict[str, Asset] = field(default_factory=dict)
    cameras: dict[str, Camera] = field(default_factory=dict)

    def To_dict(self) -> dict:
        return {
            "name": self.name,
            "assets": {
                key: {
                    "asset_type": type(asset).__name__,
                    "data": asset.To_dict()
                }
                for key, asset in self.assets.items()
            },
            "cameras": {
                key: cam.To_dict() for key, cam in self.cameras.items()},
        }

    @classmethod
    def From_dict(cls, data: dict) -> Scene:
        _assets = {}
        for _k, _i in data.get("assets", {}).items():
            _asset_class = ASSET_CLASS_REGISTRY.get(_i["asset_type"])
            if _asset_class:
                _assets[_k] = _asset_class.From_dict(_i["data"])

        return cls(
            name=data["name"],
            assets=_assets,
            cameras=dict((
                _k, Camera.From_dict(_v)
            ) for _k, _v in data.get("cameras", {}).items())
        )
