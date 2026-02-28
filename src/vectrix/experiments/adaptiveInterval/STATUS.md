# adaptiveInterval — 적응형 예측 구간 연구

## 연구 목표
시계열 특성에 따라 예측 구간 폭을 동적으로 조정하는 시스템. 현재 고정 1.96*sigma → 이질적 분산, 레짐 변화 대응.

## 실험 현황

| ID | 실험명 | 상태 | 핵심 결과 |
|----|--------|------|----------|
| E025 | 잔차 이질 분산 검출 + 적응형 구간 | 완료 | GARCH 구간 Winkler +47%, 87% 이질분산 검출 |
| E026 | Conformal vs Bootstrap vs GARCH 비교 | 완료 | GARCH 73.1% 승률, Conformal/Bootstrap 실용성 부족 |
| E027 | GARCH z-보정 Coverage 달성 | 완료 (보류) | z 보정 가능하나 GARCH Winkler 우위 미재현 |

## 채택 후보

1. **GARCH 후처리 구간** (E025, E026 공동 확인)
   - Winkler: GARCH 1431 vs 고정 2072 (+31% 개선)
   - 승률: 272/372건 (73.1%), 모든 6개 모델에서 60%+ 승률
   - 약점: Coverage 0.809 (under-coverage 15%) → z를 2.2~2.3으로 보정 필요
   - 구현 복잡도 낮음: 잔차에 GARCH(1,1) 적합 → 조건부 분산 예측

## 기각된 아이디어

1. **EWMA 적응형 구간 (E025)**: 구간 폭 과대 (3415 vs 2592), Winkler 악화 -25%
2. **Conformal Split (E026)**: Coverage 완벽(1.0)이나 구간 폭 3140으로 비실용적
3. **Conformal Jackknife (E026)**: 구간 폭 14609 — 극단적. max residual 기반 스코어 문제
4. **Bootstrap (E026)**: Coverage 0.309 (심각한 under-coverage). 잔차 재표본 감쇠 로직 문제

## 핵심 인사이트

- **이질적 분산은 보편적**: 87.4% (325/372건)에서 검출. 고정 구간 가정 위반
- **GARCH가 종합 최적**: "좁은 구간 + 변동성 적응"의 최적 균형
- 고정 구간은 Coverage 정확도(0.954)에서 최고 — 95% 목표에 가장 가까움
- **E027 추가 인사이트**: GARCH 파라미터(alpha=0.1, beta=0.85) 고정 시 데이터별 편차 큼
  - E025-E026은 rolling OOS 잔차 → GARCH 우위, E027은 in-sample 잔차 → Fixed 우위
  - GARCH 파라미터 최적화 또는 rolling CV 잔차 계산이 핵심
- **실질적 최적 전략**: GARCH 후처리 + 파라미터 최적화 + z 보정

## 다음 단계

- GARCH(1,1) 파라미터 최적화 실험 (alpha, beta, omega를 MLE로)
- Rolling CV 기반 잔차 계산으로 E027 재실험
- GARCH 후처리 모듈을 vectrix 엔진 predict()에 통합
- Conformal/Bootstrap 구현 개선 후 재비교

## 날짜
- 시작: 2026-02-28
- 최종 업데이트: 2026-02-28
