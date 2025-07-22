


from __future__ import annotations
from typing import Type, ClassVar
from dataclasses import dataclass, field
from pathlib import Path

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from plyfile import PlyData, PlyElement

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
class Gaussian_3D(Asset):
    """3D Gaussian Splatting 모델의 파라미터를 관리하는 Asset."""
    relative_path: str # 데이터 소스 폴더 기준 상대 경로 (npz, ply)
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
    )  # x, y, z, w
    # 3DGS의 SH(Spherical Harmonics) 계수 또는 일반 RGB 값으로 활용 가능.
    # SH features는 (N, sh_degree**2, 3) 이지만, 편의상 2D로 저장/로드
    # 예: SH degree 3 (16) * 3 channels = 48
    colors: NDArray = field(
        default_factory=lambda: np.empty((0, 48), dtype=np.float32),
        repr=False
    )

    def To_dict(self) -> dict:
        return {"relative_path": self.relative_path}

    @classmethod
    def From_dict(cls, data: dict) -> Gaussian_3D:
        return cls(relative_path=data["relative_path"])

    # --- 데이터 처리 로직 ---
    def __Convert_to_weight(self, display_data: dict[str, NDArray]):
        """디스플레이용 데이터를 학습용 가중치(원본 파라미터)로 변환하여 객체에 저장."""
        self.points = display_data['points']
        self.rotations = display_data['rotations']
        self.colors = display_data['colors']

        # Logit (Sigmoid의 역연산)
        _ep = 1e-10
        _op_clamped = np.clip(display_data['opacities'], _ep, 1 - _ep)
        self.opacities = -np.log(1 / _op_clamped - 1)

        # Log (Exp의 역연산)
        self.scales = np.log(display_data['scales'])

    def __Ready_to_display(self) -> dict[str, NDArray]:
        """객체의 학습용 가중치를 디스플레이용 최종 값으로 변환."""
        return {
            "points": self.points,
            "opacities": 1 / (1 + np.exp(-self.opacities)),  # Sigmoid
            "scales": np.exp(self.scales),  # Exp
            "rotations": self.rotations,
            "colors": self.colors
        }

    # --- 파일 포맷별 순수 I/O 함수 ---
    def __Load_from_npz(self, file_path: Path) -> dict[str, NDArray]:
        """npz 파일에서 디스플레이용 데이터를 읽어 dict로 반환."""
        with np.load(file_path) as data:
            return dict(data)

    def __Save_to_npz(self, file_path: Path, data: dict[str, NDArray]):
        """주어진 디스플레이용 데이터를 npz 파일로 저장."""
        np.savez(file_path, **data)

    def __Load_from_ply(self, file_path: Path) -> dict[str, NDArray]:
        """PLY 파일에서 디스플레이용 데이터를 읽어 dict로 반환."""
        _data = PlyData.read(file_path)
        _vtx: PlyElement = _data['vertex']

        _pts = np.vstack([_vtx['x'], _vtx['y'], _vtx['z']]).T
        _rots = np.vstack([
            _vtx['rot_0'], _vtx['rot_1'], _vtx['rot_2'], _vtx['rot_3']
        ]).T
        _rots /= np.linalg.norm(_rots, axis=1, keepdims=True)

        _prop_names = [
            p.name for p in _vtx.properties if p.name.startswith("f_rest_")]
        _sh_ac_ns = sorted(
            _prop_names,
            key=lambda name: int(name.split('_')[-1])
        )
        _sh_ac = np.zeros((_pts.shape[0], len(_sh_ac_ns)), dtype=np.float32)
        for i, name in enumerate(_sh_ac_ns):
            _sh_ac[:, i] = _vtx[name]

        # 최종 SH 계수 배열 생성 (N, 48)
        _clrs = np.zeros((_pts.shape[0], 48), dtype=np.float32)
        _sh_combined = np.concatenate([
            np.vstack([_vtx['f_dc_0'], _vtx['f_dc_1'], _vtx['f_dc_2']]).T,
            _sh_ac
        ], axis=1)
        _clrs[:, :_sh_combined.shape[1]] = _sh_combined

        return {
            "points": _pts,
            "opacities": _vtx['opacity'][..., None],
            "scales": np.vstack(
                [_vtx['scale_0'], _vtx['scale_1'], _vtx['scale_2']]).T,
            "rotations": _rots,
            "colors": _clrs
        }

    def __Save_to_ply(self, file_path: Path, data: dict[str, NDArray]):
        """주어진 디스플레이용 데이터를 PLY 파일로 저장."""
        _pts, _opcts, _scls, _rots, _clrs = (
            data["points"], data["opacities"], data["scales"],
            data["rotations"], data["colors"]
        )
        _n_pts = _pts.shape[0]

        # PLY 엘리먼트의 데이터 타입 정의
        _dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]

        # SH AC(고차항) 이름 동적 추가
        for i in range(_clrs.shape[1] - 3):
            _dtype_list.append((f'f_rest_{i}', 'f4'))

        _ele = np.empty(_n_pts, dtype=_dtype_list)

        # 속성 배열을 순서에 맞게 하나로 합침
        attrs = np.concatenate(
            (
                _pts, np.zeros_like(_pts),
                _clrs[:, :3],
                _opcts,
                _scls,
                _rots,
                _clrs[:, 3:]
            ),
            axis=1
        )

        _ele[:] = [tuple(row) for row in attrs]
        PlyData([PlyElement.describe(_ele, 'vertex')]).write(file_path)

    # --- 메인 데이터 입출력 컨트롤러 ---
    def Load_data(self, data_path: Path):
        """파일에서 디스플레이용 데이터를 읽어와 학습용 가중치로 변환 후 객체에 저장."""
        _f_path = data_path / self.relative_path
        _ext = _f_path.suffix

        if _ext == ".npz":
            _dp_data = self.__Load_from_npz(_f_path)
        elif _ext == ".ply":
            _dp_data = self.__Load_from_ply(_f_path)
        else:
            raise ValueError(f"지원하지 않는 파일 확장자입니다: {_ext}")

        self.__Convert_to_weight(_dp_data)

    def Save_data(self, data_path: Path):
        """객체의 학습용 가중치를 디스플레이용 데이터로 변환하여 파일에 저장."""
        if self.points.shape[0] == 0:
            return

        _f_path = data_path / self.relative_path
        _f_path.parent.mkdir(exist_ok=True, parents=True)
        _ext = _f_path.suffix

        _dp_data = self.__Ready_to_display()

        if _ext == ".npz":
            self.__Save_to_npz(_f_path, _dp_data)
        elif _ext == ".ply":
            self.__Save_to_ply(_f_path, _dp_data)
        else:
            raise ValueError(f"지원하지 않는 파일 확장자입니다: {_ext}")


def Create_test_3DGS(relative_path="test_data.ply") -> Gaussian_3D:
    """
    테스트와 동일한 데이터를 생성하여 Gaussian_3D 객체를 반환합니다.
    """
    gau_xyz = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ], dtype=np.float32)

    gau_rot = np.array([
        [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]
    ], dtype=np.float32)

    gau_s = np.array([
        [0.03, 0.03, 0.03], [0.2, 0.03, 0.03],
        [0.03, 0.2, 0.03], [0.03, 0.03, 0.2]
    ], dtype=np.float32)

    gau_a = np.array([[1], [1], [1], [1]], dtype=np.float32)

    # 기본 RGB 색상 정의 (SH 0차 계수로 사용될 값)
    gau_c_rgb = np.array([
        [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ], dtype=np.float32)

    # 셰이더 활성화 함수의 역연산을 적용하여 SH 0차 계수(DC) 생성
    C0 = 0.28209479177387814
    sh_dc = (gau_c_rgb - 0.5) / C0

    # colors 필드(N, 48) 형태에 맞게 나머지 SH 계수(AC)를 0으로 채움
    num_points = gau_xyz.shape[0]
    sh_ac_padding = np.zeros((num_points, 45), dtype=np.float32)
    sh_features = np.concatenate([sh_dc, sh_ac_padding], axis=1)

    return Gaussian_3D(
        relative_path=relative_path,
        points=gau_xyz,
        rotations=gau_rot,
        scales=gau_s,
        opacities=gau_a,
        colors=sh_features
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
        "Gaussian_3DGS": Gaussian_3D,
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
