"""
Model Registry — Single source of truth for all forecasting models.

Adding a new model:
1. Implement fit(y) -> self, predict(steps) -> (pred, lower, upper)
2. Add one entry to MODEL_REGISTRY below
3. Done. vectrix.py, types.py, docs all read from here.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a forecasting model."""

    modelId: str
    name: str
    description: str
    factory: Callable
    needsPeriod: bool = False
    minData: int = 10
    flatResistance: float = 0.5
    bestFor: tuple = ()
    tier: int = 3
    seasonal: bool = False
    hourly: bool = False


def _buildRegistry() -> Dict[str, ModelSpec]:
    """Build the model registry with lazy imports to avoid circular dependencies."""
    from .arima import AutoARIMA
    from .baselines import MeanModel, NaiveModel, RandomWalkDrift, SeasonalNaiveModel, WindowAverage
    from .ces import AutoCES
    from .croston import AutoCroston
    from .decomposition import MSTLDecomposition
    from .dot import DynamicOptimizedTheta
    from .dtsf import DynamicTimeScanForecaster
    from .esn import EchoStateForecaster
    from .ets import AutoETS, ETSModel
    from .fourTheta import AdaptiveThetaEnsemble
    from .garch import EGARCHModel, GARCHModel, GJRGARCHModel
    from .mstl import AutoMSTL
    from .tbats import AutoTBATS
    from .theta import OptimizedTheta

    specs = [
        ModelSpec(
            modelId='dot',
            name='Dynamic Optimized Theta',
            description='Joint Theta+alpha+drift optimization.',
            factory=lambda period: DynamicOptimizedTheta(period=period),
            needsPeriod=True, minData=10, flatResistance=0.80,
            bestFor=('trending data', 'general purpose'),
            tier=1, seasonal=False, hourly=True,
        ),
        ModelSpec(
            modelId='auto_ces',
            name='AutoCES (Native)',
            description='Complex exponential smoothing auto selection.',
            factory=lambda period: AutoCES(period=period),
            needsPeriod=True, minData=20, flatResistance=0.65,
            bestFor=('nonlinear patterns', 'complex seasonality'),
            tier=1, seasonal=True, hourly=True,
        ),
        ModelSpec(
            modelId='four_theta',
            name='4Theta Ensemble',
            description='Weighted combination of 4 theta lines. Top stability.',
            factory=lambda period: AdaptiveThetaEnsemble(period=period),
            needsPeriod=True, minData=10, flatResistance=0.80,
            bestFor=('general purpose', 'trending data', 'stable forecasting'),
            tier=1, seasonal=False, hourly=True,
        ),
        ModelSpec(
            modelId='auto_ets',
            name='AutoETS (Native)',
            description='Self-implemented automatic exponential smoothing.',
            factory=lambda period: AutoETS(period=period),
            needsPeriod=True, minData=20, flatResistance=0.55,
            bestFor=('stable patterns', 'short-term forecasting'),
            tier=1, seasonal=True, hourly=True,
        ),
        ModelSpec(
            modelId='auto_mstl',
            name='AutoMSTL (Native)',
            description='Auto multiple seasonality decomposition + ARIMA.',
            factory=lambda period: AutoMSTL(),
            needsPeriod=False, minData=50, flatResistance=0.85,
            bestFor=('multiple seasonality', 'complex patterns'),
            tier=1, seasonal=True, hourly=True,
        ),
        ModelSpec(
            modelId='dtsf',
            name='Dynamic Time Scan',
            description='Non-parametric pattern matching. Best ensemble diversity.',
            factory=lambda period: DynamicTimeScanForecaster(),
            needsPeriod=False, minData=30, flatResistance=0.65,
            bestFor=('repeating patterns', 'hourly data', 'seasonal data'),
            tier=2, seasonal=True, hourly=True,
        ),
        ModelSpec(
            modelId='esn',
            name='Echo State Network',
            description='Reservoir Computing nonlinear forecasting. Top accuracy.',
            factory=lambda period: EchoStateForecaster(),
            needsPeriod=False, minData=20, flatResistance=0.70,
            bestFor=('nonlinear patterns', 'regime switching', 'high volatility'),
            tier=2, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='auto_arima',
            name='AutoARIMA (Native)',
            description='Self-implemented automatic ARIMA.',
            factory=lambda period: AutoARIMA(maxP=3, maxD=2, maxQ=3),
            needsPeriod=False, minData=30, flatResistance=0.60,
            bestFor=('stationary data', 'trend forecasting'),
            tier=2, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='theta',
            name='Theta (Native)',
            description='Self-implemented Theta model.',
            factory=lambda period: OptimizedTheta(period=period),
            needsPeriod=True, minData=10, flatResistance=0.75,
            bestFor=('general purpose', 'fast forecasting'),
            tier=2, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='tbats',
            name='TBATS (Native)',
            description='Trigonometric Seasonal, Box-Cox, ARMA, Trend, Damping.',
            factory=lambda period: AutoTBATS(periods=[period]),
            needsPeriod=True, minData=30, flatResistance=0.85,
            bestFor=('multiple seasonality', 'hourly data', 'complex patterns'),
            tier=2, seasonal=True, hourly=True,
        ),
        ModelSpec(
            modelId='mstl',
            name='MSTL (Native)',
            description='Multiple seasonal decomposition.',
            factory=lambda period: MSTLDecomposition(periods=[period]),
            needsPeriod=True, minData=50, flatResistance=0.85,
            bestFor=('multiple seasonality', 'complex patterns'),
            tier=2, seasonal=True, hourly=False,
        ),
        ModelSpec(
            modelId='ets_aan',
            name='ETS(A,A,N)',
            description="Holt's Linear (additive trend, no season).",
            factory=lambda period: ETSModel('A', 'A', 'N', period),
            needsPeriod=True, minData=10, flatResistance=0.50,
            bestFor=('trending data',),
            tier=2, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='ets_aaa',
            name='ETS(A,A,A)',
            description='Holt-Winters additive seasonality.',
            factory=lambda period: ETSModel('A', 'A', 'A', period),
            needsPeriod=True, minData=20, flatResistance=0.50,
            bestFor=('seasonal data',),
            tier=2, seasonal=True, hourly=False,
        ),
        ModelSpec(
            modelId='seasonal_naive',
            name='Seasonal Naive (Native)',
            description='Seasonal naive baseline.',
            factory=lambda period: SeasonalNaiveModel(period=period),
            needsPeriod=True, minData=14, flatResistance=0.95,
            bestFor=('strong seasonality', 'high flat risk'),
            tier=2, seasonal=True, hourly=False,
        ),
        ModelSpec(
            modelId='garch',
            name='GARCH(1,1)',
            description='Conditional variance model for financial volatility.',
            factory=lambda period: GARCHModel(),
            needsPeriod=False, minData=50, flatResistance=0.50,
            bestFor=('financial data', 'volatility forecasting', 'returns'),
            tier=3, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='egarch',
            name='EGARCH',
            description='Asymmetric volatility model with leverage effect.',
            factory=lambda period: EGARCHModel(),
            needsPeriod=False, minData=50, flatResistance=0.50,
            bestFor=('financial data', 'asymmetric volatility'),
            tier=3, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='gjr_garch',
            name='GJR-GARCH',
            description='Threshold asymmetric GARCH for negative shock response.',
            factory=lambda period: GJRGARCHModel(),
            needsPeriod=False, minData=50, flatResistance=0.50,
            bestFor=('financial data', 'asymmetric volatility'),
            tier=3, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='croston',
            name='Croston (Auto)',
            description='Intermittent demand auto selection.',
            factory=lambda period: AutoCroston(),
            needsPeriod=False, minData=10, flatResistance=0.30,
            bestFor=('intermittent demand', 'zero-inflated series'),
            tier=3, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='naive',
            name='Naive',
            description='Random Walk — last value repetition.',
            factory=lambda period: NaiveModel(),
            needsPeriod=False, minData=2, flatResistance=0.10,
            bestFor=('benchmark', 'random walk data'),
            tier=3, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='mean',
            name='Mean',
            description='Historical mean forecast.',
            factory=lambda period: MeanModel(),
            needsPeriod=False, minData=2, flatResistance=0.05,
            bestFor=('benchmark', 'stationary series'),
            tier=3, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='rwd',
            name='Random Walk with Drift',
            description='Last value + average trend.',
            factory=lambda period: RandomWalkDrift(),
            needsPeriod=False, minData=5, flatResistance=0.60,
            bestFor=('trending data', 'benchmark'),
            tier=3, seasonal=False, hourly=False,
        ),
        ModelSpec(
            modelId='window_avg',
            name='Window Average',
            description='Recent window average forecast.',
            factory=lambda period: WindowAverage(window=min(period, 30)),
            needsPeriod=True, minData=5, flatResistance=0.15,
            bestFor=('benchmark', 'stable data'),
            tier=3, seasonal=False, hourly=False,
        ),
    ]

    return {spec.modelId: spec for spec in specs}


_REGISTRY_CACHE: Optional[Dict[str, ModelSpec]] = None


def getRegistry() -> Dict[str, ModelSpec]:
    """Get the model registry (cached after first call)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _buildRegistry()
    return _REGISTRY_CACHE


def getModelSpec(modelId: str) -> Optional[ModelSpec]:
    """Get a single model specification by ID."""
    return getRegistry().get(modelId)


def listModelIds() -> List[str]:
    """List all registered model IDs."""
    return list(getRegistry().keys())


def createModel(modelId: str, period: int = 1) -> Any:
    """Create a model instance from registry."""
    spec = getModelSpec(modelId)
    if spec is None:
        raise ValueError(f"Unknown model ID: '{modelId}'")
    if spec.factory is None:
        return None
    return spec.factory(period)


def getCoreModelIds() -> List[str]:
    """Get tier 1 model IDs for ensemble core pool."""
    return [mid for mid, spec in getRegistry().items() if spec.tier == 1]


def selectModels(
    n: int,
    period: int = 1,
    freq: str = 'D',
    seasonalStrength: float = 0.0,
    hasMultiSeason: bool = False,
    highFlatRisk: bool = False,
    maxModels: int = 5,
) -> List[str]:
    """Select models dynamically from registry based on data characteristics."""
    registry = getRegistry()
    isHourly = freq == 'H'
    isSeasonal = hasMultiSeason or seasonalStrength > 0.4

    candidates = []
    for mid, spec in registry.items():
        if n < spec.minData:
            continue
        if spec.tier > 2:
            continue
        if isHourly and not spec.hourly:
            continue
        candidates.append(spec)

    if isSeasonal and n >= 60:
        candidates.sort(key=lambda s: (s.tier, 0 if s.seasonal else 1))
    elif highFlatRisk:
        candidates.sort(key=lambda s: (s.tier, -s.flatResistance))
    else:
        candidates.sort(key=lambda s: s.tier)

    selected = [s.modelId for s in candidates[:maxModels]]

    if not selected:
        selected = ['dot', 'auto_ces']

    return selected


def getModelInfo() -> Dict[str, Dict]:
    """Generate MODEL_INFO dict from registry (backward compatible)."""
    registry = getRegistry()
    info = {}
    for modelId, spec in registry.items():
        info[modelId] = {
            'name': spec.name,
            'description': spec.description,
            'flatResistance': spec.flatResistance,
            'bestFor': list(spec.bestFor),
            'minData': spec.minData,
        }
    return info
