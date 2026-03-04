---
title: "한국 회귀분석"
---

# 한국 회귀분석

실제 한국 데이터셋을 활용한 두 가지 회귀분석: 서울 자전거 대여 수요 (UCI ML Repository)와 한국 거시경제 회귀 (FRED).

## 데이터 출처

| 데이터셋 | 출처 | 관측치 수 | 설명 |
|---------|------|:---------:|------|
| 서울 자전거 대여 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) | 8,760 (시간별) | 기상 조건별 시간당 자전거 대여 수 |
| 한국 거시경제 | [FRED](https://fred.stlouisfed.org/) | ~300 (월간) | 환율, 금리, 무역수지, CPI |

## 서울 자전거 대여 수요

기상 조건이 서울의 시간별 자전거 대여 수요에 미치는 영향을 분석합니다.

### 데이터 로드 및 준비

```python
import pandas as pd
from vectrix import regress

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
bikeDf = pd.read_csv(url, encoding="unicode_escape")

colMap = {
    "Rented Bike Count": "rentedBikeCount",
    "Hour": "hour",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "windSpeed",
    "Rainfall(mm)": "rainfall",
    "Solar Radiation (MJ/m2)": "solarRadiation",
}

tempCols = [c for c in bikeDf.columns if "Temperature" in c and "Dew" not in c]
if tempCols:
    colMap[tempCols[0]] = "temperature"

bikeDf = bikeDf.rename(columns=colMap)
```

### 회귀분석 실행

```python
result = regress(
    data=bikeDf,
    formula="rentedBikeCount ~ temperature + humidity + windSpeed + rainfall + hour + solarRadiation",
    summary=False,
)

print(f"R-squared:     {result.r_squared:.4f}")
print(f"Adj R-squared: {result.adj_r_squared:.4f}")
print(f"F-statistic:   {result.f_stat:.2f}")
```

### 계수 테이블

```python
labels = ["절편", "기온", "습도", "풍속", "강수량", "시간", "일사량"]

for name, coef, pval in zip(labels, result.coefficients, result.pvalues):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"{name:8s}  {coef:+10.3f}  p={pval:.4f} {sig}")
```

### 해석

- **기온** +1도 상승 시 자전거 대여 증가 (따뜻할수록 외부 활동 증가)
- **습도** +1% 상승 시 자전거 대여 감소 (불쾌한 환경)
- **강수량** +1mm 증가 시 대여 급감 (비가 오면 자전거 이용 기피)

### 진단 검정

```python
print(result.diagnose())
```

Durbin-Watson(자기상관), Breusch-Pagan(이분산성), VIF(다중공선성), Jarque-Bera(정규성) 검정을 수행합니다.

## 한국 거시경제 회귀

거시경제 요인을 활용한 원/달러 환율 모델링입니다.

### FRED 데이터 수집

```python
import pandas as pd
from vectrix import regress

series = {
    "EXKOUS": "exchangeRate",
    "INTDSRKRM193N": "interestRate",
    "XTEXVA01KRM667S": "exports",
    "KORCPIALLMINMEI": "cpi",
}

frames = []
for code, name in series.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    tmp = pd.read_csv(url)
    tmp.columns = ["date", name]
    tmp["date"] = pd.to_datetime(tmp["date"])
    frames.append(tmp.set_index("date"))

macroDf = pd.concat(frames, axis=1).dropna().reset_index()
```

### 회귀분석 실행

```python
result = regress(
    data=macroDf,
    formula="exchangeRate ~ interestRate + exports + cpi",
    summary=False,
)

print(f"R-squared:     {result.r_squared:.4f}")
print(f"Adj R-squared: {result.adj_r_squared:.4f}")
print(f"F-statistic:   {result.f_stat:.2f}")
print(result.summary())
```

### 진단 검정

```python
print(result.diagnose())
```

> **참고:** 수준 데이터에 대한 거시경제 회귀는 자기상관과 비정상성을 보이는 경우가 많습니다. 실무 분석에서는 차분이나 공적분 검정을 고려하세요. 이 쇼케이스는 API 시연 목적이며, 계량경제학 모범 사례가 아닙니다.

> **참고:** 교육 목적으로만 제공됩니다. 데이터 출처: [UCI ML Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), [FRED](https://fred.stlouisfed.org/).
