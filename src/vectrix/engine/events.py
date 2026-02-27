"""
이벤트/공휴일 효과 모듈

시계열 예측에 공휴일 및 이벤트 효과를 반영:
- 한국 공휴일 기본 내장 (양력/음력 기반)
- 사용자 정의 이벤트 추가
- 이벤트 전후 효과 범위(window) 설정
- 각 이벤트의 효과 크기 추정 (회귀 기반)

순수 numpy/scipy만 사용 (extractBatch 제외)
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

# ─── 한국 공휴일 정의 ─────────────────────────────────────────────────────

# 양력 공휴일: (월, 일, 이름)
_KR_FIXED_HOLIDAYS = [
    (1, 1, '신정'),
    (3, 1, '삼일절'),
    (5, 5, '어린이날'),
    (6, 6, '현충일'),
    (8, 15, '광복절'),
    (10, 3, '개천절'),
    (10, 9, '한글날'),
    (12, 25, '성탄절'),
]

# 음력 기반 공휴일의 대략적 양력 범위
# (실제 날짜는 매년 다르므로, 범위로 처리)
# 설날: 음력 1/1 전후 → 대략 양력 1월 중순 ~ 2월 중순
# 추석: 음력 8/15 전후 → 대략 양력 9월 초 ~ 10월 초
_KR_LUNAR_HOLIDAYS = [
    {
        'name': '설날',
        'monthRange': (1, 2),  # 1~2월 사이
        'dayRange': (15, 28),  # 대략적 범위
        'duration': 3,         # 연휴 일수
    },
    {
        'name': '추석',
        'monthRange': (9, 10),
        'dayRange': (1, 15),
        'duration': 3,
    },
]

# 연도별 설날/추석 실제 양력 날짜 (정확한 계산이 어려우므로 주요 연도 매핑)
_KR_LUNAR_DATES = {
    2020: {'설날': (1, 25), '추석': (10, 1)},
    2021: {'설날': (2, 12), '추석': (9, 21)},
    2022: {'설날': (2, 1), '추석': (9, 10)},
    2023: {'설날': (1, 22), '추석': (9, 29)},
    2024: {'설날': (2, 10), '추석': (9, 17)},
    2025: {'설날': (1, 29), '추석': (10, 6)},
    2026: {'설날': (2, 17), '추석': (9, 25)},
    2027: {'설날': (2, 7), '추석': (9, 15)},
    2028: {'설날': (1, 27), '추석': (10, 3)},
    2029: {'설날': (2, 13), '추석': (9, 22)},
    2030: {'설날': (2, 3), '추석': (9, 12)},
}


class EventEffect:
    """
    이벤트/공휴일 효과 모델

    시계열 예측에 공휴일 및 사용자 정의 이벤트의 효과를 반영.
    이벤트 전후 효과 범위를 설정하여 점진적 영향을 모델링.

    Parameters
    ----------
    holidays : str
        사용할 공휴일 세트 ('kr': 한국, 'none': 없음)
    customEvents : List[Dict], optional
        사용자 정의 이벤트 목록.
        각 이벤트는 {'name': str, 'dates': List[str], 'priorWindow': int, 'postWindow': int}
        dates 형식: 'YYYY-MM-DD' 또는 'MM-DD' (매년 반복)

    Examples
    --------
    >>> ee = EventEffect(holidays='kr')
    >>> dates = np.array(['2024-01-01', '2024-01-02', ...], dtype='datetime64')
    >>> features = ee.getEventFeatures(dates)
    """

    def __init__(
        self,
        holidays: str = 'kr',
        customEvents: Optional[List[Dict]] = None
    ):
        self.holidays = holidays.lower() if isinstance(holidays, str) else 'none'
        self.customEvents = customEvents or []
        self._eventRegistry = {}  # name -> List[dict with date info]

        # 공휴일 등록
        if self.holidays == 'kr':
            self._registerKoreanHolidays()

        # 사용자 정의 이벤트 등록
        for event in self.customEvents:
            self._registerCustomEvent(event)

    def _registerKoreanHolidays(self):
        """한국 공휴일을 이벤트 레지스트리에 등록"""
        for month, day, name in _KR_FIXED_HOLIDAYS:
            self._eventRegistry[name] = {
                'type': 'fixed',
                'month': month,
                'day': day,
                'priorWindow': 1,
                'postWindow': 1,
            }

        # 음력 공휴일
        for lunarHoliday in _KR_LUNAR_HOLIDAYS:
            self._eventRegistry[lunarHoliday['name']] = {
                'type': 'lunar',
                'monthRange': lunarHoliday['monthRange'],
                'dayRange': lunarHoliday['dayRange'],
                'duration': lunarHoliday['duration'],
                'priorWindow': 2,
                'postWindow': 1,
            }

    def _registerCustomEvent(self, event: Dict):
        """사용자 정의 이벤트 등록"""
        name = event.get('name', 'custom_event')
        self._eventRegistry[name] = {
            'type': 'custom',
            'dates': event.get('dates', []),
            'priorWindow': event.get('priorWindow', 1),
            'postWindow': event.get('postWindow', 1),
        }

    def getKoreanHolidays(self, year: int) -> List[Dict]:
        """
        특정 연도의 한국 공휴일 목록 반환

        Parameters
        ----------
        year : int
            연도

        Returns
        -------
        List[Dict]
            공휴일 목록. 각 항목: {'name': str, 'date': str, 'type': str}
        """
        holidays = []

        # 양력 공휴일
        for month, day, name in _KR_FIXED_HOLIDAYS:
            try:
                d = date(year, month, day)
                holidays.append({
                    'name': name,
                    'date': d.isoformat(),
                    'type': 'fixed',
                })
            except ValueError:
                continue

        # 음력 공휴일 (매핑 테이블에서 조회)
        if year in _KR_LUNAR_DATES:
            for name, (month, day) in _KR_LUNAR_DATES[year].items():
                duration = 3
                for offset in range(-1, duration - 1):
                    try:
                        d = date(year, month, day) + timedelta(days=offset)
                        holidays.append({
                            'name': f'{name} ({"당일" if offset == 0 else ("전날" if offset < 0 else "다음날")})',
                            'date': d.isoformat(),
                            'type': 'lunar',
                        })
                    except ValueError:
                        continue
        else:
            # 매핑이 없으면 범위 기반 대략적 날짜
            for lunarHoliday in _KR_LUNAR_HOLIDAYS:
                name = lunarHoliday['name']
                mStart, mEnd = lunarHoliday['monthRange']
                # 범위의 중앙 날짜를 대략적으로 사용
                midMonth = (mStart + mEnd) // 2 if mStart != mEnd else mStart
                midDay = 15
                try:
                    d = date(year, midMonth, midDay)
                    holidays.append({
                        'name': f'{name} (추정)',
                        'date': d.isoformat(),
                        'type': 'lunar_estimated',
                    })
                except ValueError:
                    continue

        return sorted(holidays, key=lambda x: x['date'])

    def getEventFeatures(
        self,
        dates: np.ndarray,
        periods: Optional[int] = None
    ) -> np.ndarray:
        """
        날짜 배열에 대한 이벤트 특성 행렬 생성

        Parameters
        ----------
        dates : np.ndarray
            날짜 배열 (datetime64 또는 문자열)
        periods : int, optional
            미래 기간 수 (예측용). None이면 dates 길이만큼만.

        Returns
        -------
        np.ndarray
            이벤트 특성 행렬 (n_dates, n_events)
            각 열은 하나의 이벤트에 대한 효과 (0~1 범위)
        """
        # 날짜를 datetime 객체로 변환
        parsedDates = self._parseDates(dates)
        n = len(parsedDates)

        if len(self._eventRegistry) == 0:
            return np.zeros((n, 1))

        eventNames = list(self._eventRegistry.keys())
        nEvents = len(eventNames)
        features = np.zeros((n, nEvents))

        for col, name in enumerate(eventNames):
            eventInfo = self._eventRegistry[name]
            priorWindow = eventInfo.get('priorWindow', 1)
            postWindow = eventInfo.get('postWindow', 1)

            for i, dt in enumerate(parsedDates):
                effect = self._computeEventEffect(dt, eventInfo, priorWindow, postWindow)
                features[i, col] = effect

        return features

    def _computeEventEffect(
        self,
        dt: date,
        eventInfo: Dict,
        priorWindow: int,
        postWindow: int
    ) -> float:
        """
        특정 날짜에 대한 이벤트 효과 계산

        이벤트 당일 = 1.0, 전후 기간은 거리에 따라 감소 (선형).
        """
        eventType = eventInfo['type']

        if eventType == 'fixed':
            return self._fixedHolidayEffect(dt, eventInfo, priorWindow, postWindow)
        elif eventType == 'lunar':
            return self._lunarHolidayEffect(dt, eventInfo, priorWindow, postWindow)
        elif eventType == 'custom':
            return self._customEventEffect(dt, eventInfo, priorWindow, postWindow)
        return 0.0

    def _fixedHolidayEffect(
        self,
        dt: date,
        eventInfo: Dict,
        priorWindow: int,
        postWindow: int
    ) -> float:
        """양력 고정 공휴일의 효과"""
        try:
            holidayDate = date(dt.year, eventInfo['month'], eventInfo['day'])
        except ValueError:
            return 0.0

        diff = (dt - holidayDate).days

        if diff == 0:
            return 1.0
        elif -priorWindow <= diff < 0:
            # 이벤트 전: 가까울수록 효과 큼
            return 1.0 - abs(diff) / (priorWindow + 1)
        elif 0 < diff <= postWindow:
            # 이벤트 후: 가까울수록 효과 큼
            return 1.0 - diff / (postWindow + 1)
        return 0.0

    def _lunarHolidayEffect(
        self,
        dt: date,
        eventInfo: Dict,
        priorWindow: int,
        postWindow: int
    ) -> float:
        """음력 기반 공휴일의 효과"""
        year = dt.year

        # 정확한 날짜가 매핑 테이블에 있으면 사용
        for lunarH in _KR_LUNAR_HOLIDAYS:
            if (eventInfo.get('monthRange') == lunarH['monthRange'] and
                    eventInfo.get('dayRange') == lunarH['dayRange']):
                name = lunarH['name']
                break
        else:
            return 0.0

        if year in _KR_LUNAR_DATES and name in _KR_LUNAR_DATES[year]:
            month, day = _KR_LUNAR_DATES[year][name]
            try:
                holidayDate = date(year, month, day)
            except ValueError:
                return 0.0

            duration = eventInfo.get('duration', 3)
            # 연휴 기간 (-1일 ~ +1일 for 3일 연휴)
            for offset in range(-(duration // 2), (duration + 1) // 2):
                checkDate = holidayDate + timedelta(days=offset)
                diff = (dt - checkDate).days
                if diff == 0:
                    return 1.0

            # 전후 윈도우
            startDate = holidayDate - timedelta(days=duration // 2)
            endDate = holidayDate + timedelta(days=(duration + 1) // 2 - 1)
            diffToStart = (dt - startDate).days
            diffFromEnd = (dt - endDate).days

            if -priorWindow <= diffToStart < 0:
                return 1.0 - abs(diffToStart) / (priorWindow + 1)
            elif 0 < diffFromEnd <= postWindow:
                return 1.0 - diffFromEnd / (postWindow + 1)
        else:
            # 매핑 없으면 범위 기반 대략적 매칭
            mStart, mEnd = eventInfo['monthRange']
            dStart, dEnd = eventInfo['dayRange']
            if mStart <= dt.month <= mEnd and dStart <= dt.day <= dEnd:
                return 0.5  # 불확실하므로 낮은 효과
        return 0.0

    def _customEventEffect(
        self,
        dt: date,
        eventInfo: Dict,
        priorWindow: int,
        postWindow: int
    ) -> float:
        """사용자 정의 이벤트의 효과"""
        eventDates = eventInfo.get('dates', [])

        for eventDateStr in eventDates:
            try:
                eventDate = self._parseEventDate(eventDateStr, dt.year)
            except Exception:
                continue

            diff = (dt - eventDate).days

            if diff == 0:
                return 1.0
            elif -priorWindow <= diff < 0:
                return 1.0 - abs(diff) / (priorWindow + 1)
            elif 0 < diff <= postWindow:
                return 1.0 - diff / (postWindow + 1)

        return 0.0

    def _parseEventDate(self, dateStr: str, defaultYear: int) -> date:
        """이벤트 날짜 문자열 파싱"""
        parts = dateStr.split('-')
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return date(defaultYear, int(parts[0]), int(parts[1]))
        raise ValueError(f"날짜 형식 오류: {dateStr}")

    def _parseDates(self, dates: np.ndarray) -> List[date]:
        """날짜 배열을 date 객체 리스트로 변환"""
        parsed = []
        for d in dates:
            try:
                if isinstance(d, (datetime, date)):
                    parsed.append(d if isinstance(d, date) else d.date())
                elif isinstance(d, np.datetime64):
                    ts = (d - np.datetime64('1970-01-01', 'D')).astype(int)
                    parsed.append(date(1970, 1, 1) + timedelta(days=int(ts)))
                elif isinstance(d, str):
                    parts = d.split('-')
                    parsed.append(date(int(parts[0]), int(parts[1]), int(parts[2])))
                else:
                    # 숫자면 타임스탬프로 간주
                    parsed.append(datetime.fromtimestamp(float(d)).date())
            except Exception:
                # 파싱 실패 시 에포크 시작으로 대체
                parsed.append(date(1970, 1, 1))
        return parsed

    def estimateEffects(
        self,
        y: np.ndarray,
        dates: np.ndarray
    ) -> Dict:
        """
        각 이벤트의 효과 크기 추정

        이벤트 특성 행렬과 시계열 데이터를 사용하여
        최소제곱 회귀로 각 이벤트의 효과 계수를 추정.

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터
        dates : np.ndarray
            날짜 배열

        Returns
        -------
        Dict
            각 이벤트의 효과 추정치.
            {
                'eventName': {
                    'coefficient': float,  # 효과 크기
                    'significance': float, # 통계적 유의성 (t-stat)
                    'nOccurrences': int,   # 이벤트 발생 횟수
                },
                ...
                '_summary': {
                    'totalEventEffect': float,  # 전체 이벤트 효과 비율
                    'rSquared': float,           # 이벤트 모델의 설명력
                }
            }
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        features = self.getEventFeatures(dates)
        n = len(y)
        nEvents = features.shape[1]

        if n != features.shape[0]:
            raise ValueError(
                f"시계열 길이({n})와 날짜 수({features.shape[0]})가 일치하지 않습니다."
            )

        eventNames = list(self._eventRegistry.keys())
        if len(eventNames) == 0:
            return {'_summary': {'totalEventEffect': 0.0, 'rSquared': 0.0}}

        result = {}

        try:
            # 절편 포함 설계 행렬
            X = np.column_stack([np.ones(n), features])

            # 최소제곱 해 (정규방정식)
            XtX = X.T @ X
            # 정칙화 (Ridge)
            XtX += 1e-8 * np.eye(XtX.shape[0])
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)

            # 잔차 및 통계
            yHat = X @ beta
            residuals = y - yHat
            sse = np.sum(residuals ** 2)
            sst = np.sum((y - np.mean(y)) ** 2)
            rSquared = 1 - sse / max(sst, 1e-10) if sst > 0 else 0.0

            # 표준오차
            df = max(n - X.shape[1], 1)
            mse = sse / df
            try:
                varBeta = mse * np.linalg.inv(XtX)
                seBeta = np.sqrt(np.maximum(np.diag(varBeta), 0))
            except np.linalg.LinAlgError:
                seBeta = np.ones(X.shape[1]) * 1e10

            # 각 이벤트 결과
            for i, name in enumerate(eventNames):
                coeff = beta[i + 1]  # +1: 절편 스킵
                se = max(seBeta[i + 1], 1e-10)
                tStat = coeff / se
                nOccurrences = int(np.sum(features[:, i] > 0))

                result[name] = {
                    'coefficient': float(coeff),
                    'significance': float(abs(tStat)),
                    'nOccurrences': nOccurrences,
                }

            # 이벤트 효과 비율
            eventContribution = np.sum(features @ beta[1:])
            totalSignal = np.sum(np.abs(y - np.mean(y)))
            eventRatio = abs(eventContribution) / max(totalSignal, 1e-10)

            result['_summary'] = {
                'totalEventEffect': float(np.clip(eventRatio, 0, 1)),
                'rSquared': float(np.clip(rSquared, 0, 1)),
            }

        except Exception:
            # fallback: 단순 평균 비교
            for i, name in enumerate(eventNames):
                mask = features[:, i] > 0
                if np.any(mask) and np.any(~mask):
                    eventMean = np.mean(y[mask])
                    nonEventMean = np.mean(y[~mask])
                    result[name] = {
                        'coefficient': float(eventMean - nonEventMean),
                        'significance': 0.0,
                        'nOccurrences': int(np.sum(mask)),
                    }
                else:
                    result[name] = {
                        'coefficient': 0.0,
                        'significance': 0.0,
                        'nOccurrences': 0,
                    }
            result['_summary'] = {
                'totalEventEffect': 0.0,
                'rSquared': 0.0,
            }

        return result

    def addEvent(self, event: Dict):
        """
        이벤트 추가

        Parameters
        ----------
        event : Dict
            이벤트 정보. 필수: 'name', 'dates'.
            선택: 'priorWindow' (기본 1), 'postWindow' (기본 1).
        """
        self.customEvents.append(event)
        self._registerCustomEvent(event)

    def listEvents(self) -> List[str]:
        """등록된 모든 이벤트 이름 반환"""
        return list(self._eventRegistry.keys())

    def getEventCalendar(
        self,
        startDate: str,
        endDate: str
    ) -> List[Dict]:
        """
        기간 내 이벤트 캘린더 생성

        Parameters
        ----------
        startDate : str
            시작일 (YYYY-MM-DD)
        endDate : str
            종료일 (YYYY-MM-DD)

        Returns
        -------
        List[Dict]
            기간 내 이벤트 목록
        """
        try:
            start = date.fromisoformat(startDate)
            end = date.fromisoformat(endDate)
        except (ValueError, AttributeError):
            sp = startDate.split('-')
            start = date(int(sp[0]), int(sp[1]), int(sp[2]))
            ep = endDate.split('-')
            end = date(int(ep[0]), int(ep[1]), int(ep[2]))

        calendar = []
        years = range(start.year, end.year + 1)

        for year in years:
            if self.holidays == 'kr':
                holidays = self.getKoreanHolidays(year)
                for h in holidays:
                    hDate = date.fromisoformat(h['date'])
                    if start <= hDate <= end:
                        calendar.append(h)

        # 사용자 정의 이벤트
        for event in self.customEvents:
            name = event.get('name', 'custom')
            for dateStr in event.get('dates', []):
                try:
                    for year in years:
                        eDate = self._parseEventDate(dateStr, year)
                        if start <= eDate <= end:
                            calendar.append({
                                'name': name,
                                'date': eDate.isoformat(),
                                'type': 'custom',
                            })
                except Exception:
                    continue

        return sorted(calendar, key=lambda x: x['date'])
