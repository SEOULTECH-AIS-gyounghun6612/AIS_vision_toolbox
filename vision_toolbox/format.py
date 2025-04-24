from __future__ import annotations

from typing import Annotated, Any
from dataclasses import dataclass, field  # fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import inv

from vision_toolbox.utils import (
    TF, TP, INTRINSIC, VEC_2, VEC_3, VEC_4, IMGs_SIZE, PTS_COLOR,
    IMG_1C_GROUP, IMG_3C_GROUP,
    Camera_Process, Convert
)


class utils():
    @staticmethod
    def Tranpose_align(
        from_tp: TP, to_tp: TP
    ) -> Annotated[NDArray, (4, 4)]:
        assert len(from_tp) == len(to_tp)

        _tp: TP = np.matmul(to_tp, np.linalg.inv(from_tp))
        _tr = np.eye(4)

        _tr[:3, :3] = Convert.Rv_2_M(
            Convert.M_2_Rv(_tp[:, :3, :3]).mean(axis=0))
        _tr[:3, 3] = _tp[:, :3, 3].mean(axis=0)

        return _tr


class Vision_Object():
    @dataclass
    class Basement():
        def Load(self, path: str | Path) -> bool:
            raise NotImplementedError

        def Save(self, path: str | Path, force: bool = True) -> bool:
            raise NotImplementedError

        def Push(self, **kwarg):
            raise NotImplementedError

        def Pop(self, data_num: int | list[int]):
            raise NotImplementedError

        def Capture(self):
            raise NotImplementedError

        def To_dict(self) -> dict[str, Any]:
            raise NotImplementedError

        def From_dict(self, **data):
            raise NotImplementedError

        @staticmethod
        def Get_annotation() -> str:
            raise NotImplementedError

    @dataclass
    class Camera(Basement):
        fov: VEC_2 = field(
            default_factory=lambda: np.ones((1, 2)) * np.pi * 2 / 3
        )
        principal_rate: VEC_2 = field(
            default_factory=lambda: np.ones((1, 2)) * 0.5
        )
        distortion: VEC_4 = field(default_factory=lambda: np.zeros((1, 4)))
        img_size: IMGs_SIZE = field(
            default_factory=lambda: np.array([[480, 720]])
        )

        def Get_intrinsic(self):
            return Camera_Process.Compose_intrinsic(
                Camera_Process.Get_focal_length_from(self.fov, self.img_size),
                self.img_size
            )

        def Set_parameter(self, intrinsic: INTRINSIC):
            _f, _pp = Camera_Process.Extract_intrinsic(intrinsic)
            _img_size = self.img_size

            self.fov = Camera_Process.Get_fov_from(self.img_size, _f)
            self.principal_rate = Camera_Process.Get_pp_rate_from(
                _pp, _img_size)

        def Push(
            self,
            fov: tuple[float, float],
            principal_rate: tuple[float, float],
            distortion: tuple[float, float, float, float],
            img_size: tuple[int, int]
        ):
            self.fov = np.r_[self.fov, [fov]]
            self.principal_rate = np.r_[self.principal_rate, [principal_rate]]
            self.distortion = np.r_[self.distortion, [distortion]]
            self.img_size = np.r_[self.img_size, [img_size]]

        def To_dict(self):
            _param = np.concat(
                [self.fov, self.principal_rate, self.distortion],
                axis=1
            )
            return {
                "parameter": _param.tolist(),
                "image_size": self.img_size.tolist()
            }

        def Load(self, path: str | Path):
            ...

        def From_dict(self, **data: list):
            _parma = np.array(data["parameter"])

            self.fov = _parma[:, :2]
            self.principal_rate = _parma[:, 2:4]
            self.distortion = _parma[:, 4:]

            self.img_size = np.array(data["image_size"], dtype=np.int32)

        def Save(self, path: str | Path):
            ...

    @dataclass
    class Pose(Basement):
        Q: VEC_4 = field(
            default_factory=lambda: np.eye(1, 4)
        )  # Quaternion rotation (qw, qx, qy, qz)
        transfer: TF = field(default_factory=lambda: np.zeros((1, 3, 1)))
        # Tx, Ty, Tz

        # file_name: InitVar[str]
        # image: np.ndarray | None = field(init=False)

        def Push( self, quat: VEC_4, transfer: TF ):
            ...

        def Get_transpose(self):
            _q = self.Q
            _tp: TP = np.tile(np.eye(4), [_q.shape[0], 1, 1])
            _tp[:, :3, :3] = Convert.Q_to_M(_q)
            _tp[:, :3, 3] = self.transfer
            return _tp

        def Get_inv_transpose(self):
            return inv(self.Get_transpose())

        # def To_list(self, rx_type: Literal["euler", "rovec", "Q"] = "Q"):
        #     _r_v, _t_v = self.Get_pose_to_vector(rx_type)

        #     return ",".join([f"{_v:>.5f}" for _v in _r_v + _t_v])

    @dataclass
    class Image(Basement):
        file_name: list[str | None] = field(default_factory=list)
        img: IMG_1C_GROUP | IMG_3C_GROUP | None = None

    @dataclass
    class Points(Basement):
        file_name: str | None = None

        points: VEC_3 = field(
            default_factory=lambda: np.empty((0, 3)))  # x, y, z
        colors: PTS_COLOR = field(
            default_factory=lambda: np.empty((0, 3), dtype=np.uint8))
        # b, g, r


@dataclass
class Point_Cloud(Vision_Object):
    points: VEC_3 = field(
        default_factory=lambda: np.empty((0, 3)))  # x, y, z
    colors: PTS_COLOR = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.uint8))  # b, g, r

    # def __Get_field_list__(self):
    #     return [_c.name for _c in fields(self) if _c.name != "file_name"]

    # def __Decoding_Point_data__(self, path: Path):
    #     _field_list = self.__Get_field_list__()
    #     with load(path) as _f:
    #         for _k, _d in _f.items():
    #             if _k in _field_list:
    #                 setattr(self, _k, _d)

    def To_dict(self) -> str:
        # return f"{','.join(self.__Get_field_list__())},{self.file_name}"
        return f"{self.file_name}"

    def From_dict(self, line: list[str]):
        self.file_name = line[-1]

    def Get_annotation(self) -> list[str]:
        return ["# points (tx, ty, tz), colors (b, g, r), file_name"]


# SENSOR = TypeVar("SENSOR", bound=Vision_Object)


# @dataclass
# class Scene_Sequence(Generic[SENSOR]):
#     load_source: InitVar[bool]
#     data_format: type[SENSOR]

#     meta_file: str = "test.json"
#     file_annotation: list[str] = field(default_factory=list)
#     source_path: str = str(Path.cwd() / "data")

#     sequence_data: dict[int, SENSOR] = field(init=False)

#     def __post_init__(self, load_source: bool):
#         self.Load_data(load_source)

#     def Save_data(self, save_source: bool = True, **kwarg) -> bool:
#         _file = Path(self.meta_file)  # meta file
#         # check the save directory for meta file
#         _file.parent.mkdir(exist_ok=True)

#         # check the save source option
#         _path = Path(self.source_path)  # source_path
#         _save_source = save_source and _path.exists()

#         _sq_data = self.sequence_data
#         # check tha annotation that set in the meta file
#         _annotation = _sq_data[0].Get_annotation()
#         self.file_annotation = _annotation

#         _meta_data = []

#         for _id, _data in _sq_data.items():
#             if _save_source:
#                 _data.Save(_path)
#             _meta_data.append(f"{_id},{_data.To_dict(**kwarg)}")

#         _path.write_text("\n".join(_meta_data), encoding="UTF-8")
#         return True

#     def Load_data(self, load_source: bool = True) -> bool:
#         _file = Path(self.meta_file)  # meta file
#         _sq_data = {}

#         if not all(
#           [_file.exists(), _file.is_file(), _file.suffix == ".txt"]):
#             self.sequence_data = _sq_data
#             return False

#         _path = Path(self.source_path)  # source_path
#         _load_source = load_source and _path.exists()

#         _st = len(self.file_annotation)
#         _data_format = self.data_format

#         for _line in _file.read_text(encoding="UTF-8").split("\n")[_st:]:
#             _data = _line.split(",")
#             _sq = _data_format()
#             _sq.From_dict(*_data[1:])

#             if _load_source:
#                 _sq.Load(_path)
#             _sq_data[int(_data[0])] = _sq

#         self.sequence_data = _sq_data
#         return True

#     def Get_config(self):
#         return {
#             "data_format": self.data_format.__name__.lower(),
#             "meta_file": self.meta_file,
#             "file_annotation": self.file_annotation,
#             "path": self.source_path,
#         }
