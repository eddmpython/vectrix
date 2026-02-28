"""Tests for vectrix.datasets module."""

import numpy as np
import pandas as pd
import pytest

from vectrix.datasets import listSamples, loadSample


_ALL_NAMES = ["airline", "retail", "stock", "temperature", "energy", "web", "intermittent"]


class TestLoadSample:
    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_returnsDataFrame(self, name):
        df = loadSample(name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_hasDateColumn(self, name):
        df = loadSample(name)
        assert "date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_hasNumericValueColumn(self, name):
        df = loadSample(name)
        valueCols = [c for c in df.columns if c != "date"]
        assert len(valueCols) >= 1
        for col in valueCols:
            assert pd.api.types.is_numeric_dtype(df[col])

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_noMissingValues(self, name):
        df = loadSample(name)
        assert df.isna().sum().sum() == 0

    def test_airlineShape(self):
        df = loadSample("airline")
        assert len(df) == 144
        assert "passengers" in df.columns

    def test_retailShape(self):
        df = loadSample("retail")
        assert len(df) == 730
        assert "sales" in df.columns

    def test_stockShape(self):
        df = loadSample("stock")
        assert len(df) == 252
        assert "close" in df.columns

    def test_temperatureShape(self):
        df = loadSample("temperature")
        assert len(df) == 1095
        assert "temperature" in df.columns

    def test_energyShape(self):
        df = loadSample("energy")
        assert len(df) == 720
        assert "consumption_kwh" in df.columns

    def test_webShape(self):
        df = loadSample("web")
        assert len(df) == 180
        assert "pageviews" in df.columns

    def test_intermittentShape(self):
        df = loadSample("intermittent")
        assert len(df) == 365
        assert "demand" in df.columns
        zeroRatio = (df["demand"] == 0).mean()
        assert zeroRatio > 0.5

    def test_caseInsensitive(self):
        df1 = loadSample("Airline")
        df2 = loadSample("AIRLINE")
        assert len(df1) == len(df2)

    def test_unknownNameRaises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            loadSample("nonexistent")

    def test_deterministicOutput(self):
        df1 = loadSample("airline")
        df2 = loadSample("airline")
        pd.testing.assert_frame_equal(df1, df2)


class TestListSamples:
    def test_returnsDataFrame(self):
        result = listSamples()
        assert isinstance(result, pd.DataFrame)

    def test_allDatasetsListed(self):
        result = listSamples()
        assert len(result) == 7
        assert set(result["name"].tolist()) == set(_ALL_NAMES)

    def test_hasExpectedColumns(self):
        result = listSamples()
        for col in ["name", "description", "valueCol", "frequency", "rows"]:
            assert col in result.columns
