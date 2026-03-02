# Vectrix MCP Server

[Model Context Protocol](https://modelcontextprotocol.io/) server that exposes Vectrix time series forecasting capabilities as tools for AI assistants.

## Features

### Tools (10)
| Tool | Description |
|------|------------|
| `forecast_timeseries` | Forecast future values from numeric data |
| `forecast_csv` | Forecast from a CSV file on disk |
| `analyze_timeseries` | DNA profiling — patterns, seasonality, anomalies |
| `compare_models` | Compare all 30+ models and rank by accuracy |
| `run_regression` | R-style regression with diagnostics |
| `detect_anomalies` | Z-score, IQR, seasonal, rolling anomaly detection |
| `backtest_model` | Walk-forward cross-validation |
| `list_sample_datasets` | List built-in sample datasets |
| `load_sample_dataset` | Load a sample dataset for testing |

### Resources (2)
| Resource | Description |
|----------|------------|
| `vectrix://models` | List of all available forecasting models |
| `vectrix://api-reference` | Quick API reference |

### Prompts (2)
| Prompt | Description |
|--------|------------|
| `forecast_workflow` | Step-by-step forecasting workflow |
| `regression_workflow` | Step-by-step regression analysis workflow |

## Setup

### Prerequisites

```bash
pip install "mcp[cli]" vectrix
```

### Claude Code

```bash
claude mcp add --transport stdio vectrix -- uv run python mcp/server.py
```

Verify:
```bash
claude mcp list
```

### Claude Desktop

Add to `claude_desktop_config.json`:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
    "mcpServers": {
        "vectrix": {
            "command": "uv",
            "args": ["run", "python", "/absolute/path/to/mcp/server.py"]
        }
    }
}
```

### Testing

```bash
npx -y @modelcontextprotocol/inspector
```

## Usage Examples

Once connected, ask your AI assistant:

- "Forecast the next 12 months from this sales data: [100, 120, 130, ...]"
- "Analyze this time series for patterns and anomalies"
- "Compare all available models on my data"
- "Run a regression: revenue ~ ads + price"
- "Detect anomalies in this data with threshold 2.0"
- "Backtest the forecast accuracy with 5-fold validation"
- "Load the airline sample dataset and forecast 24 months"
