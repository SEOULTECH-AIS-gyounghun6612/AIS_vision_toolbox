# Vision toolbox in AIS

연구 과정 중 사용되는 비전 관련 기능들을 구성한 Python 기반의 도구 모음입니다.
비전 데이터의 구조 정의, 포맷 변환, 시각화 등의 기능을 통합하여 연구 및 개발 효율성을 높이는 것을 목표로 합니다.

## 🚧 Update Plan

### format.py

- [ ] 데이터 구조 변경
  - [ ] 카메라 클래스에서 상속받던 Pose 클래스 제거
  - [ ] 공간상에서 특정 시점을 표현하는 기본 구조 클래스인 Scene 클래스 추가

## 🔧 Installation

### 📚 Requirements

- [python_ex](https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git)

- scipy

### pip 사용 (HTTPS)

```bash
# using pip with https
pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_vision_toolbox.git
```

### 직접 설치 (로컬 clone)

```bash
# clone this repository
git clone https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_vision_toolbox.git

cd AIS_vision_toolbox

# install 
pip install -e .
```
