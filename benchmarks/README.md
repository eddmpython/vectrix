# Vectrix Benchmarks

M3/M4 Competition benchmarks for evaluating Vectrix forecasting accuracy.

## Quick Start

```bash
# M3 Competition (3,003 series)
python benchmarks/runM3.py --cat M3Month --n 100
python benchmarks/runM3.py --all --n 50

# M4 Competition (100,000 series)
python benchmarks/runM4.py --freq Monthly --n 100
python benchmarks/runM4.py --all --n 50
```

## Files

| File | Description |
|------|-------------|
| `m3Loader.py` | M3 data loader (forvis.github.io) |
| `m4Loader.py` | M4 data loader (GitHub/Mcompetitions) |
| `runM3.py` | M3 benchmark runner |
| `runM4.py` | M4 benchmark runner |

## Categories

**M3**: `M3Year` (6h), `M3Quart` (8h), `M3Month` (18h), `M3Other` (8h)

**M4**: `Yearly` (6h), `Quarterly` (8h), `Monthly` (18h), `Weekly` (13h), `Daily` (14h), `Hourly` (48h)

*h = forecast horizon*

## Output

Results are saved to `m3Results.csv` and `m4Results.csv` (git-ignored).
