"""
TSFrame: Time Series DataFrame

시계열 전용 DataFrame 래퍼:
- 자동 날짜/주기 감지
- 내장 forecast/decompose 메서드
- 결측치 처리, 리샘플링
- train/test split
"""

from .tsframe import TSFrame

__all__ = ["TSFrame"]
