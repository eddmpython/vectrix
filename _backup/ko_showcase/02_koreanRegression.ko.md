# 쇼케이스 02 — 한국 데이터 회귀분석

**실제 한국 데이터셋으로 진단이 포함된 회귀분석을 수행합니다.**

## 개요

공개 한국 데이터를 활용한 두 가지 회귀분석:

1. **서울 자전거 대여량** — 기상 요인으로 시간별 대여 수 예측 (UCI ML Repository)
2. **한국 거시경제 회귀** — 경제 지표를 활용한 환율 결정요인 분석 (FRED)

## 인터랙티브 실행

```bash
pip install vectrix pandas marimo
marimo run docs/showcase/ko/02_koreanRegression.py
```

## 1. 서울 자전거 대여

서울 공공 자전거의 8,760개 시간별 관측치.

```python
import pandas as pd
from vectrix import regress

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
bikeDf = pd.read_csv(url, encoding="unicode_escape")

bikeDf = bikeDf.rename(columns={
    "Rented Bike Count": "rentedBikeCount",
    "Hour": "hour",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "windSpeed",
    "Rainfall(mm)": "rainfall",
    "Solar Radiation (MJ/m2)": "solarRadiation",
})
tempCols = [c for c in bikeDf.columns if "Temperature" in c and "Dew" not in c]
if tempCols:
    bikeDf = bikeDf.rename(columns={tempCols[0]: "temperature"})

result = regress(
    data=bikeDf,
    formula="rentedBikeCount ~ temperature + humidity + windSpeed + rainfall + hour + solarRadiation",
)
```

### 주요 발견

| 변수 | 효과 |
|------|------|
| 온도 +1°C | 대여량 증가 |
| 습도 +1% | 대여량 감소 |
| 강수량 +1mm | 대여량 크게 감소 |
| 시간대 | 출퇴근 시간 피크 |

```python
print(f"R-squared: {result.r_squared:.4f}")
print(f"F-statistic: {result.f_stat:.2f}")
print(result.diagnose())
```

## 2. 한국 거시경제 회귀

FRED 경제 지표를 활용한 환율 결정요인 분석.

```python
import pandas as pd
from vectrix import regress

series = {
    "EXKOUS": "exchangeRate",
    "IRLTLT01KRM156N": "bondYield",
    "LRUNTTTTMKM156S": "unemployment",
    "SPASTT01KRM661N": "kospi",
    "KORCPIALLMINMEI": "cpi",
}

dfs = []
for code, name in series.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    df = pd.read_csv(url)
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"])
    dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on="date", how="inner")
merged = merged.dropna()

result = regress(
    data=merged,
    formula="exchangeRate ~ bondYield + unemployment + kospi + cpi",
)
```

## 데이터 출처

| 출처 | 데이터셋 | 인증 |
|------|---------|:----:|
| UCI ML Repository | 서울 자전거 대여량 | 불필요 |
| FRED | 한국 경제 지표 | 불필요 |

!!! note "주의사항"
    교육 목적으로만 사용하세요.

---

**다음:** [쇼케이스 03 — 모델 비교](03_modelComparison.md)
