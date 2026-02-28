# frequencyDomain — 주파수 도메인 예측 연구

## 연구 목표
FFT/웨이블릿 기반 잔차 보정으로 M4 Daily/Hourly OWA 개선. 시간 도메인 모델이 놓치는 주기 패턴을 주파수 도메인에서 포착.

## 실험 현황

| ID | 실험명 | 상태 | 핵심 결과 |
|----|--------|------|----------|
| E023 | FFT 잔차 보정 | 완료 (기각) | 승률 30.6%, 과적합 문제 |
| E024 | Spectral Denoising | 완료 (조건부 채택) | Seasonal 58.3% 승률, adaptive 60% |
| E028 | FFT Hybrid Forecaster | 완료 (조건부 채택) | 평균 순위 4.77, hourly/volatile 1위 |

## 채택 후보

1. **Spectral Denoising (E024)**
   - 조건부 적용: 계절성 + 변동성 데이터 한정
   - 에너지 임계값 80%가 최적
   - auto_ces (adaptive 69.2%), dot (59.0%)에서 가장 효과적
   - DNA 특성으로 계절성 강도 판단 후 조건부 적용
   - Trending/Stationary에서는 비활성화 필수

2. **FFT Hybrid Forecaster (E028)**
   - 새 모델 3종 창조: FourierForecaster, DampedFourierForecaster, AdaptiveFourierForecaster
   - FFT 분해: trend (선형) + periodic (top-K 주파수) + residual (AR(1))
   - 평균 순위 4.77 (theta 5.23보다 우위)
   - hourlyMultiSeasonal 1위 (34% 개선), stockPrice 1위 (40% 개선), volatile 1위
   - 조건부 적용: hourly/다중 주기/비추세 데이터 한정
   - Trending 데이터에서 약점 (추세 모델링이 선형으로 제한)

## 기각된 아이디어

1. **FFT 잔차 보정 (E023)**: 승률 30.6%, 과적합 심각
   - 잔차 FFT 성분 ≠ 미래 반복 패턴 (과거 노이즈 재투영)
   - Energy ratio가 높을수록 과보정 (상관 -0.23)
   - MSTL처럼 이미 계절성 잘 잡는 모델은 보정 효과 최저

## 핵심 인사이트

- **잔차 보정 vs 입력 정제**: 잔차에 FFT 보정 추가는 과적합, 입력 데이터 정제(denoising)는 효과적
- Spectral denoising은 "신호 대 잡음비" 개선 → 모델 학습 품질 향상
- multiSeasonalRetail에서 5/5 전승 — 다중 계절성 데이터에서 강력
- volatile 데이터도 5/5 전승 — 고주파 노이즈 제거 효과
- Trending에서 심각한 악화 — 추세 정보까지 손실되는 문제
- **FFT 기반 새 모델은 기존 통계 모델과 다른 예측 패턴**: 앙상블 후보로서 가치 있음
- FourierForecaster는 주기성이 강한 데이터에서 기존 모델을 압도하지만 추세 데이터에서 약함

## 다음 단계

- Spectral denoising을 vectrix 전처리 파이프라인에 조건부 통합
- DNA 특성(seasonalStrength, spectralEntropy) 기반 on/off 로직
- 추세 보존 개선: linear detrend → 더 정교한 추세 추출

## 날짜
- 시작: 2026-02-28
- 최종 업데이트: 2026-02-28
