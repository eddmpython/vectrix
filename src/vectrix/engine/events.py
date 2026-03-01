"""
Event/Holiday Effect Module

Incorporates holiday and event effects into time series forecasting:
- Built-in Korean holidays (solar/lunar calendar based)
- User-defined custom event support
- Pre/post event effect window configuration
- Event effect size estimation (regression based)

Pure numpy/scipy only (except extractBatch)
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

_US_FIXED_HOLIDAYS = [
    (1, 1, "New Year's Day"),
    (7, 4, "Independence Day"),
    (11, 11, "Veterans Day"),
    (12, 25, "Christmas Day"),
]

_US_FLOATING_HOLIDAYS = [
    {"name": "MLK Day", "month": 1, "weekday": 0, "week": 3},
    {"name": "Presidents' Day", "month": 2, "weekday": 0, "week": 3},
    {"name": "Memorial Day", "month": 5, "weekday": 0, "week": -1},
    {"name": "Labor Day", "month": 9, "weekday": 0, "week": 1},
    {"name": "Columbus Day", "month": 10, "weekday": 0, "week": 2},
    {"name": "Thanksgiving", "month": 11, "weekday": 3, "week": 4},
]

_JP_FIXED_HOLIDAYS = [
    (1, 1, "元日"),
    (2, 11, "建国記念の日"),
    (2, 23, "天皇誕生日"),
    (4, 29, "昭和の日"),
    (5, 3, "憲法記念日"),
    (5, 4, "みどりの日"),
    (5, 5, "こどもの日"),
    (7, 20, "海の日"),
    (8, 11, "山の日"),
    (9, 23, "秋分の日"),
    (10, 14, "スポーツの日"),
    (11, 3, "文化の日"),
    (11, 23, "勤労感謝の日"),
]

_CN_FIXED_HOLIDAYS = [
    (1, 1, "元旦"),
    (5, 1, "劳动节"),
    (10, 1, "国庆节"),
    (10, 2, "国庆节"),
    (10, 3, "国庆节"),
]


def _nthWeekdayOfMonth(year: int, month: int, weekday: int, n: int) -> date:
    """Get the nth weekday of a month (n=-1 for last)."""
    if n == -1:
        if month == 12:
            lastDay = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            lastDay = date(year, month + 1, 1) - timedelta(days=1)
        offset = (lastDay.weekday() - weekday) % 7
        return lastDay - timedelta(days=offset)
    else:
        firstDay = date(year, month, 1)
        offset = (weekday - firstDay.weekday()) % 7
        return firstDay + timedelta(days=offset + 7 * (n - 1))


# ─── Korean Holiday Definitions ───────────────────────────────────────────

_KR_FIXED_HOLIDAYS = [
    (1, 1, 'New Year'),
    (3, 1, 'Independence Movement Day'),
    (5, 5, "Children's Day"),
    (6, 6, 'Memorial Day'),
    (8, 15, 'Liberation Day'),
    (10, 3, 'National Foundation Day'),
    (10, 9, 'Hangul Day'),
    (12, 25, 'Christmas'),
]

# Approximate solar date ranges for lunar-based holidays
# (Actual dates vary each year, so handled as ranges)
# Seollal: Lunar 1/1 vicinity -> approx. mid-Jan to mid-Feb in solar calendar
# Chuseok: Lunar 8/15 vicinity -> approx. early Sep to early Oct in solar calendar
_KR_LUNAR_HOLIDAYS = [
    {
        'name': 'Seollal',
        'monthRange': (1, 2),
        'dayRange': (15, 28),
        'duration': 3,
    },
    {
        'name': 'Chuseok',
        'monthRange': (9, 10),
        'dayRange': (1, 15),
        'duration': 3,
    },
]

# Actual solar dates for Seollal/Chuseok by year (mapped since exact lunar calculation is complex)
_KR_LUNAR_DATES = {
    2020: {'Seollal': (1, 25), 'Chuseok': (10, 1)},
    2021: {'Seollal': (2, 12), 'Chuseok': (9, 21)},
    2022: {'Seollal': (2, 1), 'Chuseok': (9, 10)},
    2023: {'Seollal': (1, 22), 'Chuseok': (9, 29)},
    2024: {'Seollal': (2, 10), 'Chuseok': (9, 17)},
    2025: {'Seollal': (1, 29), 'Chuseok': (10, 6)},
    2026: {'Seollal': (2, 17), 'Chuseok': (9, 25)},
    2027: {'Seollal': (2, 7), 'Chuseok': (9, 15)},
    2028: {'Seollal': (1, 27), 'Chuseok': (10, 3)},
    2029: {'Seollal': (2, 13), 'Chuseok': (9, 22)},
    2030: {'Seollal': (2, 3), 'Chuseok': (9, 12)},
}


class EventEffect:
    """
    Event/Holiday Effect Model

    Incorporates holiday and user-defined event effects into time series forecasting.
    Models gradual impact by configuring pre/post event effect windows.

    Parameters
    ----------
    holidays : str
        Holiday set to use ('kr': Korea, 'us': US, 'jp': Japan, 'cn': China, 'none': none)
    customEvents : List[Dict], optional
        List of user-defined events.
        Each event: {'name': str, 'dates': List[str], 'priorWindow': int, 'postWindow': int}
        Date format: 'YYYY-MM-DD' or 'MM-DD' (recurring annually)

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

        _registrationMap = {
            'kr': self._registerKoreanHolidays,
            'us': self._registerUSHolidays,
            'jp': self._registerJPHolidays,
            'cn': self._registerCNHolidays,
        }
        registerFn = _registrationMap.get(self.holidays)
        if registerFn:
            registerFn()

        for event in self.customEvents:
            self._registerCustomEvent(event)

    def _registerKoreanHolidays(self):
        """Register Korean holidays in the event registry"""
        for month, day, name in _KR_FIXED_HOLIDAYS:
            self._eventRegistry[name] = {
                'type': 'fixed',
                'month': month,
                'day': day,
                'priorWindow': 1,
                'postWindow': 1,
            }

        # Lunar holidays
        for lunarHoliday in _KR_LUNAR_HOLIDAYS:
            self._eventRegistry[lunarHoliday['name']] = {
                'type': 'lunar',
                'monthRange': lunarHoliday['monthRange'],
                'dayRange': lunarHoliday['dayRange'],
                'duration': lunarHoliday['duration'],
                'priorWindow': 2,
                'postWindow': 1,
            }

    def _registerUSHolidays(self):
        for month, day, name in _US_FIXED_HOLIDAYS:
            self._eventRegistry[name] = {
                'type': 'fixed',
                'month': month,
                'day': day,
                'priorWindow': 1,
                'postWindow': 1,
            }
        for fh in _US_FLOATING_HOLIDAYS:
            self._eventRegistry[fh['name']] = {
                'type': 'floating',
                'month': fh['month'],
                'weekday': fh['weekday'],
                'week': fh['week'],
                'priorWindow': 1,
                'postWindow': 1,
            }

    def _registerJPHolidays(self):
        for month, day, name in _JP_FIXED_HOLIDAYS:
            self._eventRegistry[name] = {
                'type': 'fixed',
                'month': month,
                'day': day,
                'priorWindow': 1,
                'postWindow': 1,
            }

    def _registerCNHolidays(self):
        for month, day, name in _CN_FIXED_HOLIDAYS:
            self._eventRegistry[name] = {
                'type': 'fixed',
                'month': month,
                'day': day,
                'priorWindow': 1,
                'postWindow': 1,
            }

    def _registerCustomEvent(self, event: Dict):
        """Register a user-defined custom event"""
        name = event.get('name', 'custom_event')
        self._eventRegistry[name] = {
            'type': 'custom',
            'dates': event.get('dates', []),
            'priorWindow': event.get('priorWindow', 1),
            'postWindow': event.get('postWindow', 1),
        }

    def getKoreanHolidays(self, year: int) -> List[Dict]:
        """
        Return Korean holiday list for a specific year

        Parameters
        ----------
        year : int
            Year

        Returns
        -------
        List[Dict]
            Holiday list. Each item: {'name': str, 'date': str, 'type': str}
        """
        holidays = []

        # Solar calendar holidays
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

        # Lunar holidays (lookup from mapping table)
        if year in _KR_LUNAR_DATES:
            for name, (month, day) in _KR_LUNAR_DATES[year].items():
                duration = 3
                for offset in range(-1, duration - 1):
                    try:
                        d = date(year, month, day) + timedelta(days=offset)
                        holidays.append({
                            'name': f'{name} ({"day of" if offset == 0 else ("day before" if offset < 0 else "day after")})',
                            'date': d.isoformat(),
                            'type': 'lunar',
                        })
                    except ValueError:
                        continue
        else:
            # If no mapping exists, use approximate range-based dates
            for lunarHoliday in _KR_LUNAR_HOLIDAYS:
                name = lunarHoliday['name']
                mStart, mEnd = lunarHoliday['monthRange']
                # Use approximate midpoint of the range
                midMonth = (mStart + mEnd) // 2 if mStart != mEnd else mStart
                midDay = 15
                try:
                    d = date(year, midMonth, midDay)
                    holidays.append({
                        'name': f'{name} (estimated)',
                        'date': d.isoformat(),
                        'type': 'lunar_estimated',
                    })
                except ValueError:
                    continue

        return sorted(holidays, key=lambda x: x['date'])

    def getHolidays(self, year: int) -> List[Dict]:
        """
        Return holiday list for the currently configured country

        Parameters
        ----------
        year : int
            Year

        Returns
        -------
        List[Dict]
            Holiday list. Each item: {'name': str, 'date': str, 'type': str}
        """
        if self.holidays == 'kr':
            return self.getKoreanHolidays(year)

        holidays = []
        fixedMap = {
            'us': _US_FIXED_HOLIDAYS,
            'jp': _JP_FIXED_HOLIDAYS,
            'cn': _CN_FIXED_HOLIDAYS,
        }
        fixedList = fixedMap.get(self.holidays, [])
        for month, day, name in fixedList:
            holidays.append({
                'name': name,
                'date': date(year, month, day).isoformat(),
                'type': 'fixed',
            })

        if self.holidays == 'us':
            for fh in _US_FLOATING_HOLIDAYS:
                d = _nthWeekdayOfMonth(year, fh['month'], fh['weekday'], fh['week'])
                holidays.append({
                    'name': fh['name'],
                    'date': d.isoformat(),
                    'type': 'floating',
                })

        return sorted(holidays, key=lambda x: x['date'])

    def adjustForecast(
        self,
        predictions: np.ndarray,
        futureDates: np.ndarray,
        effects: Dict,
    ) -> np.ndarray:
        """
        Apply estimated event effects to forecast values

        Parameters
        ----------
        predictions : np.ndarray
            Original prediction array
        futureDates : np.ndarray
            Date array for the forecast period (datetime64 or string)
        effects : Dict
            Return value from estimateEffects()

        Returns
        -------
        np.ndarray
            Predictions with event effects applied
        """
        predictions = np.asarray(predictions, dtype=np.float64).copy()
        features = self.getEventFeatures(futureDates)
        eventNames = list(self._eventRegistry.keys())

        for col, name in enumerate(eventNames):
            if name in effects and name != '_summary':
                coeff = effects[name].get('coefficient', 0.0)
                predictions += features[:, col] * coeff

        return predictions

    def getEventFeatures(
        self,
        dates: np.ndarray,
        periods: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate event feature matrix for a date array

        Parameters
        ----------
        dates : np.ndarray
            Date array (datetime64 or string)
        periods : int, optional
            Number of future periods (for forecasting). If None, uses length of dates.

        Returns
        -------
        np.ndarray
            Event feature matrix (n_dates, n_events)
            Each column represents the effect of one event (range 0~1)
        """
        # Convert dates to datetime objects
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
        Compute event effect for a specific date

        Event day = 1.0, pre/post periods decay linearly with distance.
        """
        eventType = eventInfo['type']

        if eventType == 'fixed':
            return self._fixedHolidayEffect(dt, eventInfo, priorWindow, postWindow)
        elif eventType == 'floating':
            return self._floatingHolidayEffect(dt, eventInfo, priorWindow, postWindow)
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
        """Effect of a fixed solar calendar holiday"""
        try:
            holidayDate = date(dt.year, eventInfo['month'], eventInfo['day'])
        except ValueError:
            return 0.0

        diff = (dt - holidayDate).days

        if diff == 0:
            return 1.0
        elif -priorWindow <= diff < 0:
            # Pre-event: closer means larger effect
            return 1.0 - abs(diff) / (priorWindow + 1)
        elif 0 < diff <= postWindow:
            # Post-event: closer means larger effect
            return 1.0 - diff / (postWindow + 1)
        return 0.0

    def _floatingHolidayEffect(
        self,
        dt: date,
        eventInfo: Dict,
        priorWindow: int,
        postWindow: int
    ) -> float:
        month = eventInfo['month']
        weekday = eventInfo['weekday']
        week = eventInfo['week']
        holidayDate = _nthWeekdayOfMonth(dt.year, month, weekday, week)
        diff = (dt - holidayDate).days

        if diff == 0:
            return 1.0
        elif -priorWindow <= diff < 0:
            return 1.0 - abs(diff) / (priorWindow + 1)
        elif 0 < diff <= postWindow:
            return 1.0 - diff / (postWindow + 1)
        return 0.0

    def _lunarHolidayEffect(
        self,
        dt: date,
        eventInfo: Dict,
        priorWindow: int,
        postWindow: int
    ) -> float:
        """Effect of a lunar calendar based holiday"""
        year = dt.year

        # Use exact date if available in the mapping table
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
            # Holiday period (-1 day ~ +1 day for 3-day holiday)
            for offset in range(-(duration // 2), (duration + 1) // 2):
                checkDate = holidayDate + timedelta(days=offset)
                diff = (dt - checkDate).days
                if diff == 0:
                    return 1.0

            # Pre/post windows
            startDate = holidayDate - timedelta(days=duration // 2)
            endDate = holidayDate + timedelta(days=(duration + 1) // 2 - 1)
            diffToStart = (dt - startDate).days
            diffFromEnd = (dt - endDate).days

            if -priorWindow <= diffToStart < 0:
                return 1.0 - abs(diffToStart) / (priorWindow + 1)
            elif 0 < diffFromEnd <= postWindow:
                return 1.0 - diffFromEnd / (postWindow + 1)
        else:
            # Approximate range-based matching if no mapping exists
            mStart, mEnd = eventInfo['monthRange']
            dStart, dEnd = eventInfo['dayRange']
            if mStart <= dt.month <= mEnd and dStart <= dt.day <= dEnd:
                return 0.5  # Low effect due to uncertainty
        return 0.0

    def _customEventEffect(
        self,
        dt: date,
        eventInfo: Dict,
        priorWindow: int,
        postWindow: int
    ) -> float:
        """Effect of a user-defined custom event"""
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
        """Parse event date string"""
        parts = dateStr.split('-')
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return date(defaultYear, int(parts[0]), int(parts[1]))
        raise ValueError(f"Invalid date format: {dateStr}")

    def _parseDates(self, dates: np.ndarray) -> List[date]:
        """Convert date array to list of date objects"""
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
                    # If numeric, treat as timestamp
                    parsed.append(datetime.fromtimestamp(float(d)).date())
            except Exception:
                # Fallback to epoch start on parse failure
                parsed.append(date(1970, 1, 1))
        return parsed

    def estimateEffects(
        self,
        y: np.ndarray,
        dates: np.ndarray
    ) -> Dict:
        """
        Estimate effect size of each event

        Uses event feature matrix and time series data to estimate
        each event's effect coefficient via least squares regression.

        Parameters
        ----------
        y : np.ndarray
            Time series data
        dates : np.ndarray
            Date array

        Returns
        -------
        Dict
            Effect estimates for each event.
            {
                'eventName': {
                    'coefficient': float,  # Effect size
                    'significance': float, # Statistical significance (t-stat)
                    'nOccurrences': int,   # Number of event occurrences
                },
                ...
                '_summary': {
                    'totalEventEffect': float,  # Overall event effect ratio
                    'rSquared': float,           # Explanatory power of event model
                }
            }
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        features = self.getEventFeatures(dates)
        n = len(y)
        nEvents = features.shape[1]

        if n != features.shape[0]:
            raise ValueError(
                f"Time series length ({n}) does not match date count ({features.shape[0]})."
            )

        eventNames = list(self._eventRegistry.keys())
        if len(eventNames) == 0:
            return {'_summary': {'totalEventEffect': 0.0, 'rSquared': 0.0}}

        result = {}

        try:
            # Design matrix with intercept
            X = np.column_stack([np.ones(n), features])

            # Least squares solution (normal equations)
            XtX = X.T @ X
            # Regularization (Ridge)
            XtX += 1e-8 * np.eye(XtX.shape[0])
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)

            # Residuals and statistics
            yHat = X @ beta
            residuals = y - yHat
            sse = np.sum(residuals ** 2)
            sst = np.sum((y - np.mean(y)) ** 2)
            rSquared = 1 - sse / max(sst, 1e-10) if sst > 0 else 0.0

            # Standard errors
            df = max(n - X.shape[1], 1)
            mse = sse / df
            try:
                varBeta = mse * np.linalg.inv(XtX)
                seBeta = np.sqrt(np.maximum(np.diag(varBeta), 0))
            except np.linalg.LinAlgError:
                seBeta = np.ones(X.shape[1]) * 1e10

            # Per-event results
            for i, name in enumerate(eventNames):
                coeff = beta[i + 1]  # +1: skip intercept
                se = max(seBeta[i + 1], 1e-10)
                tStat = coeff / se
                nOccurrences = int(np.sum(features[:, i] > 0))

                result[name] = {
                    'coefficient': float(coeff),
                    'significance': float(abs(tStat)),
                    'nOccurrences': nOccurrences,
                }

            # Event effect ratio
            eventContribution = np.sum(features @ beta[1:])
            totalSignal = np.sum(np.abs(y - np.mean(y)))
            eventRatio = abs(eventContribution) / max(totalSignal, 1e-10)

            result['_summary'] = {
                'totalEventEffect': float(np.clip(eventRatio, 0, 1)),
                'rSquared': float(np.clip(rSquared, 0, 1)),
            }

        except Exception:
            # fallback: simple mean comparison
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
        Add an event

        Parameters
        ----------
        event : Dict
            Event information. Required: 'name', 'dates'.
            Optional: 'priorWindow' (default 1), 'postWindow' (default 1).
        """
        self.customEvents.append(event)
        self._registerCustomEvent(event)

    def listEvents(self) -> List[str]:
        """Return all registered event names"""
        return list(self._eventRegistry.keys())

    def getEventCalendar(
        self,
        startDate: str,
        endDate: str
    ) -> List[Dict]:
        """
        Generate event calendar within a date range

        Parameters
        ----------
        startDate : str
            Start date (YYYY-MM-DD)
        endDate : str
            End date (YYYY-MM-DD)

        Returns
        -------
        List[Dict]
            List of events within the date range
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
            if self.holidays != 'none':
                holidays = self.getHolidays(year)
                for h in holidays:
                    hDate = date.fromisoformat(h['date'])
                    if start <= hDate <= end:
                        calendar.append(h)

        # User-defined events
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
