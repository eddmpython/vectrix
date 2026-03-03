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

Rust 엔진이 **wheel에 내장**되어 있습니다 — 별도 설치, 컴파일러 불필요. Polars처럼 설치만 하면 바로 빠릅니다.

## 선택적 추가 기능

```bash
pip install vectrix                  # 30+ 모델 + Rust 엔진 내장
pip install "vectrix[ml]"            # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[foundation]"    # + Amazon Chronos-2, Google TimesFM 2.5
pip install "vectrix[all]"           # 전체
```

## 내장 Rust 엔진

25개 핵심 예측 핫 루프가 Rust로 가속되어 모든 wheel에 컴파일됩니다:

- Linux (x86_64, manylinux)
- macOS (Apple Silicon ARM + x86_64)
- Windows (x86_64)
- Python 3.10, 3.11, 3.12, 3.13

Rust 컴파일러 불필요. 코드 변경 없이 설치만 하면 자동으로 빨라집니다.

| 구성요소 | Python Only | Rust 적용 | 속도 향상 |
|:---------|:-----------|:----------|:---------|
| `forecast()` 200pts | 295ms | **52ms** | **5.6x** |
| AutoETS fit | 348ms | **32ms** | **10.8x** |
| DOT fit | 240ms | **10ms** | **24x** |
| ETS filter (핫 루프) | 0.17ms | **0.003ms** | **67x** |

## 설치 확인

```python
import vectrix
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140], steps=3)
print(result.predictions)
```

Rust 엔진 활성화 확인:

```python
print(vectrix.TURBO_AVAILABLE)  # True
```

## 핵심 의존성

Vectrix는 3개의 필수 의존성만 있습니다:

| 패키지 | 최소 버전 |
|--------|----------|
| numpy | >= 1.24.0 |
| pandas | >= 2.0.0 |
| scipy | >= 1.10.0 |
