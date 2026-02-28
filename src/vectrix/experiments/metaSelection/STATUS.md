# metaSelection — DNA 메타러닝 모델 선택 연구

## 연구 목표
DNA 65+ 시계열 특성으로 최적 예측 모델을 자동 선택하는 메타러닝 시스템 구축

## 실험 현황

| ID | 실험명 | 상태 | 핵심 결과 |
|----|--------|------|----------|
| E012 | DNA 메타러닝 검증 | 완료 | top-3 hit 38.9%, mstl inf 버그 발견/수정 |
| E013 | 특성 중요도 분석 | 완료 | Ridge LOO-CV 50.0%, Top-5 특성 식별 |
| E014 | 앙상블 가중치 | 완료 (기각) | 앙상블보다 top-1 단독이 우수 (36.78% vs 59%) |
| E015 | 조건부 2-모델 전략 | 완료 (부분 채택) | Gap=신뢰도 지표(corr=0.42), 실전 개선 1% |
| E016 | Ridge 통합 검증 | 완료 (보류) | Ridge 압도적 우위, 그러나 과적합 50% |

## 채택된 변경사항 (모듈 코드 반영 완료)

1. **AutoMSTL inf/nan 버그 수정** → `engine/mstl.py`
   - predict()에 inf 가드, _detectPeriods 임계값 강화
2. **DNA 추천 규칙 리밸런싱** → `adaptive/dna.py`
   - mstl 점수 하향, auto_ces/dot 점수 상향, 데이터 길이 조건 추가

## 보류 중 (과적합 해소 후 통합 예정)

1. **Ridge 메타모델로 _recommendModels() 교체**
   - 성능: Ridge top-3 50% vs 규칙 기반 21% (+29%p)
   - MAPE: Ridge 30.02% vs 규칙 73.32% (2.4x 차이)
   - 승률: Ridge 55승 vs 규칙 2승 (61전)
   - 추론: 0.004ms (규칙 0.005ms보다 빠름), 내장 7.1KB
   - **문제: train 100% vs LOO-CV 50% = 과적합 갭 50%**
   - **원인: 62개 학습 데이터가 12 클래스 분류에 부족 (클래스당 5.2개)**
   - 해결 방안: M4 데이터로 학습셋 확대 or 규칙+Ridge 하이브리드

## 기각된 아이디어

1. **DNA 가중 앙상블 (E014)**: 5개 모델 가중 결합이 top-1 단독보다 나쁨.
   극단적 MAPE 모델이 평균을 오염시킴. 앙상블 가치는 "선택"에 있지 "결합"에 있지 않음.
2. **조건부 2-모델 전략 (E015)**: Gap 신뢰도 지표는 유효(corr=0.42)하나,
   실전 MAPE 개선이 1.0%에 불과. 복잡도 대비 이득 미미.
3. **즉시 Ridge 통합 (E016)**: 과적합 갭 50%로 사전학습 계수 직접 내장은 시기상조.

## 핵심 인사이트

- **특성 중요도 Top-5**: volatilityClustering, seasonalPeakPeriod, nonlinearAutocorr, demandDensity, hurstExponent
- seasonalStrength/trendStrength는 Top-10에도 없음 → 기존 규칙 기반의 근본 한계
- Ridge 메타모델(LOO-CV 50%)이 규칙 기반(21%)보다 +29%p 우수
- Ridge MAPE(30.02%)가 Oracle(29.97%)과 거의 동일 (사전학습 기준)
- Oracle 모델 분포: mstl 21%, auto_arima 14.5%, tbats 12.9% (매우 분산됨)
- **Gap 신뢰도**: gap > 0.49이면 정답률 73.3%, gap < 0.09이면 6.2%
- Alpha=0.01이 LOO-CV 최적 (top-3=53.2%, top-5=69.4%)
- 앙상블은 "결합"이 아닌 "선택"이 가치

## 다음 단계

- M4 데이터셋으로 학습 데이터 확대 → 과적합 해소
- 규칙 기반 + Ridge 하이브리드 전략 검증
- 또는 다른 연구 방향(ensembleEvolution, adaptiveInterval) 착수

## 날짜
- 시작: 2026-02-28
- 최종 업데이트: 2026-02-28
