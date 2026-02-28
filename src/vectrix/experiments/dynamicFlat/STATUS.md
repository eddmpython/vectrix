# dynamicFlat — 동적 Flat Defense 연구

## 연구 목표
현재 고정 임계값 기반 flat defense를 스펙트럼 분석 기반으로 동적 전환. 시계열 특성에 따라 flat 판정 기준을 자동 조정.

## 실험 현황

| ID | 실험명 | 상태 | 핵심 결과 |
|----|--------|------|----------|
| E021 | 스펙트럼 기반 flat 임계값 | 완료 (기각) | FP Rate 88% — 동적 임계값도 해결 불가 |
| E022 | Horizon-Aware Flat Detection | 완료 (보류) | 최고 F1=0.645 (기존), 새 기준 모두 열위 |

## 채택된 변경사항
(아직 없음)

## 기각된 아이디어

1. **스펙트럼 동적 임계값 (E021)**: FP Rate 88% → 90%. 근본 원인은 임계값이 아니라 판단 기준 자체
   - 365일 원본 std vs 14일 예측 std 직접 비교가 구조적 결함
   - 모든 정상 예측이 원본 대비 변동성이 극도로 작음

2. **Window-based std ratio (E022)**: FPR 0.880 변화 없음
3. **Seasonal Correlation (E022)**: FPR 0.800 소폭 개선만
4. **Diff Entropy (E022)**: FPR 0.400으로 절반 감소했으나 Recall 0.250
5. **Combined (E022)**: FPR 0.800, F1 0.431로 기존보다 나쁨

## 핵심 인사이트

- **현재 Flat Defense의 근본 결함 발견**: FP Rate 88%
  - 거의 모든 정상 예측을 flat으로 오감지
  - theta, dot, arima의 14일 예측은 본질적으로 추세만 반영 → flat과 구분 불가
  - MSTL만 계절성 반영하여 정상 판정 가능
- **"Flat ≠ Bad" 패러다임**: stationary 데이터에서 flat 예측은 오히려 정확
- **진짜 질문 전환 필요**: "이 예측이 flat인가?" → "이 예측이 과소 변동인가?"
- **Flat Defense를 "Pattern Fidelity Score"로 재정의 필요**:
  - 학습 데이터의 계절 패턴 강도 vs 예측의 패턴 보존도 비교
  - 계절성이 강한 데이터에서만 flat 감지 활성화

## 다음 단계

- "Pattern Fidelity Score" 개념 설계 및 프로토타입
- 학습 데이터 계절 분해 → 예측에 동일 수준 패턴 보존 여부 판단
- DNA 특성(seasonalStrength)과 연동하여 flat defense on/off 자동화

## 날짜
- 시작: 2026-02-28
- 최종 업데이트: 2026-02-28
