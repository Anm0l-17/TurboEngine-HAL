"""Tests for adaptive conformal prediction."""

import numpy as np
import pytest

from src.uncertainty.adaptive_conformal import AdaptiveConformalRegressor, _weighted_quantile


class TestWeightedQuantile:
    def test_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert _weighted_quantile(values, weights, 0.5) == pytest.approx(2.5)

    def test_all_weight_on_one(self):
        values = np.array([1.0, 10.0])
        weights = np.array([1.0, 0.0])
        assert _weighted_quantile(values, weights, 0.9) == pytest.approx(1.0)

    def test_quantile_extremes(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.5, 0.3, 0.2])
        q0 = _weighted_quantile(values, weights, 0.0)
        q1 = _weighted_quantile(values, weights, 1.0)
        assert q0 == pytest.approx(1.0)
        assert q1 == pytest.approx(3.0)


class TestAdaptiveConformalRegressor:
    def test_init_invalid_coverage(self):
        with pytest.raises(ValueError, match="coverage"):
            AdaptiveConformalRegressor(coverage=0.0)
        with pytest.raises(ValueError, match="coverage"):
            AdaptiveConformalRegressor(coverage=1.0)

    def test_predict_before_fit_raises(self):
        model = AdaptiveConformalRegressor()
        with pytest.raises(RuntimeError, match="call fit"):
            model.predict_interval(np.array([[1.0]]), np.array([[0.5]]))

    def test_fit_and_predict_interval(self):
        rng = np.random.default_rng(42)
        cal_features = rng.normal(size=(100, 5))
        cal_truth = rng.normal(size=100)
        cal_pred = cal_truth + rng.normal(size=100) * 0.1

        model = AdaptiveConformalRegressor(coverage=0.8, n_neighbours=20)
        model.fit(cal_features, cal_truth, cal_pred)

        test_features = rng.normal(size=(10, 5))
        test_pred = rng.normal(size=10)
        lower, upper, emp_cov = model.predict_interval(test_features, test_pred)

        assert lower.shape == (10,)
        assert upper.shape == (10,)
        assert (lower <= upper + 1e-12).all()
        assert 0 <= emp_cov <= 1

    def test_multi_target(self):
        rng = np.random.default_rng(42)
        cal_features = rng.normal(size=(50, 3))
        cal_truth = rng.normal(size=(50, 2))
        cal_pred = cal_truth + rng.normal(size=(50, 2)) * 0.1

        model = AdaptiveConformalRegressor(coverage=0.9)
        model.fit(cal_features, cal_truth, cal_pred)

        test_features = rng.normal(size=(5, 3))
        test_pred = rng.normal(size=(5, 2))
        lower, upper, emp_cov = model.predict_interval(test_features, test_pred)

        assert lower.shape == (5, 2)
        assert upper.shape == (5, 2)
        assert (lower <= upper + 1e-12).all()

    def test_fit_returns_self(self):
        model = AdaptiveConformalRegressor()
        cal = np.ones((10, 2))
        result = model.fit(cal, np.ones(10), np.ones(10))
        assert result is model
