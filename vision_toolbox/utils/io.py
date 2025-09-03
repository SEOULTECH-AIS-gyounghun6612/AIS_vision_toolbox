"""에셋 및 씬 데이터 로딩/저장을 위한 I/O 함수 모음."""

import json
from pathlib import Path
import numpy as np
import cv2

from ..asset import Scene, Image, Point_Cloud, Gaussian_3DGS

__all__ = [
    "Load_scene", "Load_image", "Load_point_cloud", "Load_gaussian_3dgs",
    "Load_all_asset_data", "Save_scene", "Save_image", "Save_point_cloud",
    "Save_gaussian_3dgs", "Save_all_asset_data"
]


def Load_scene(file_path: Path) -> Scene:
    """JSON 파일로부터 Scene 객체를 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as _f:
        _data = json.load(_f)
    return Scene.From_dict(_data)


def Load_image(image_asset: Image, data_root: Path):
    """Image asset의 이미지 데이터를 .data 필드로 로드합니다."""
    _img_path = data_root / image_asset.relative_path
    if not _img_path.exists():
        print(f"Warning: Image file not found at {_img_path}")
        image_asset.data = None
        return
    _bgr_img = cv2.imread(str(_img_path))
    image_asset.data = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2RGB)


def Load_point_cloud(pc_asset: Point_Cloud, data_root: Path):
    """Point_Cloud asset의 포인트 클라우드 데이터를 로드합니다."""
    _file_path = data_root / pc_asset.relative_path
    with np.load(_file_path) as _data:
        pc_asset.points = _data['points']
        pc_asset.colors = _data['colors']


def Load_gaussian_3dgs(gs_asset: Gaussian_3DGS, data_root: Path):
    """Gaussian_3DGS asset의 데이터를 로드합니다."""
    _file_path = data_root / gs_asset.relative_path
    with np.load(_file_path) as _data:
        gs_asset.points = _data['points']
        gs_asset.colors = _data['colors']
        gs_asset.opacities = _data['opacities']
        gs_asset.scales = _data['scales']
        gs_asset.rotations = _data['rotations']
        gs_asset.sh_features = _data['sh_features']


def Load_all_asset_data(scene: Scene, data_root: Path):
    """
    Scene 객체를 순회하며 모든 연관된 대용량 데이터를 메모리로 로드합니다.
    """
    print(f"Loading all asset data for scene '{scene.name}'...")
    # Scene-level 3D assets
    for _asset in scene.assets.values():
        if isinstance(_asset, Gaussian_3DGS):
            Load_gaussian_3dgs(_asset, data_root)
        elif isinstance(_asset, Point_Cloud):
            Load_point_cloud(_asset, data_root)
    
    # Camera-level 2D assets
    for _camera in scene.cameras.values():
        for _asset in _camera.assets.values():
            if isinstance(_asset, Image):
                Load_image(_asset, data_root)


def Save_scene(scene: Scene, file_path: Path):
    """Scene 객체를 JSON 파일로 직렬화합니다."""
    file_path.parent.mkdir(exist_ok=True, parents=True)
    with open(file_path, 'w', encoding='utf-8') as _f:
        json.dump(scene.To_dict(), _f, indent=4)


def Save_image(image_asset: Image, data_root: Path):
    """Image asset의 .data 필드에 있는 데이터를 파일로 저장합니다."""
    if image_asset.data is None: return
    _img_path = data_root / image_asset.relative_path
    _img_path.parent.mkdir(exist_ok=True, parents=True)
    _bgr_img = cv2.cvtColor(image_asset.data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(_img_path), _bgr_img)


def Save_point_cloud(pc_asset: Point_Cloud, data_root: Path):
    """Point_Cloud asset의 데이터를 NPZ 파일로 저장합니다."""
    _file_path = data_root / pc_asset.relative_path
    _file_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(_file_path, points=pc_asset.points, colors=pc_asset.colors)


def Save_gaussian_3dgs(gs_asset: Gaussian_3DGS, data_root: Path):
    """Gaussian_3DGS asset의 데이터를 NPZ 파일로 저장합니다."""
    _file_path = data_root / gs_asset.relative_path
    _file_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        _file_path, 
        points=gs_asset.points, 
        colors=gs_asset.colors,
        opacities=gs_asset.opacities,
        scales=gs_asset.scales,
        rotations=gs_asset.rotations,
        sh_features=gs_asset.sh_features
    )


def Save_all_asset_data(scene: Scene, data_root: Path):
    """
    Scene 객체를 순회하며 메모리에 있는 모든 연관된 대용량 데이터를 
    파일로 저장합니다.
    """
    print(f"Saving all asset data for scene '{scene.name}'...")
    # Scene-level 3D assets
    for _asset in scene.assets.values():
        if isinstance(_asset, Gaussian_3DGS):
            Save_gaussian_3dgs(_asset, data_root)
        elif isinstance(_asset, Point_Cloud):
            Save_point_cloud(_asset, data_root)
            
    # Camera-level 2D assets
    for _camera in scene.cameras.values():
        for _asset in _camera.assets.values():
            if isinstance(_asset, Image):
                Save_image(_asset, data_root)