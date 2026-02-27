# 설치

## 요구 사항

- Python 3.10+
- OS: Windows, macOS, Linux

## 설치

=== "pip"

    ```bash
    pip install vectrix
    ```

=== "uv"

    ```bash
    uv add vectrix
    ```

## 선택적 추가 기능

```bash
pip install "vectrix[numba]"       # Numba JIT 가속 (2-5배 속도 향상)
pip install "vectrix[ml]"          # LightGBM, XGBoost, scikit-learn
pip install "vectrix[tutorials]"   # 인터랙티브 marimo 튜토리얼
pip install "vectrix[all]"         # 전체
```

## 설치 확인

```python
import vectrix
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140], steps=3)
print(result.predictions)
```

## 핵심 의존성

Vectrix는 3개의 필수 의존성만 있습니다:

| 패키지 | 최소 버전 |
|--------|----------|
| numpy | >= 1.24.0 |
| pandas | >= 2.0.0 |
| scipy | >= 1.10.0 |
