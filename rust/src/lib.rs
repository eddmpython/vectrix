use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn ets_filter(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    level0: f64,
    trend0: f64,
    seasonal0: PyReadonlyArray1<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    phi: f64,
    period: usize,
    error_type: i32,
    trend_type: i32,
    seasonal_type: i32,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    Py<PyArray1<f64>>,
)> {
    let y = y.as_array();
    let n = y.len();
    let mut fitted = Array1::<f64>::zeros(n);
    let mut residuals = Array1::<f64>::zeros(n);
    let mut seasonal: Vec<f64> = seasonal0.as_array().to_vec();

    let mut level = level0;
    let mut trend = trend0;

    for t in 0..n {
        let season_idx = t % period;

        let s = match seasonal_type {
            1 => seasonal[season_idx],
            2 => {
                if seasonal[season_idx] < 1e-6 {
                    seasonal[season_idx] = 1e-6;
                }
                seasonal[season_idx]
            }
            _ => 0.0,
        };

        let yhat = match (trend_type, seasonal_type) {
            (0, 1) => level + s,
            (0, 2) => level * s,
            (0, _) => level,
            (1 | 2, 1) => level + phi * trend + s,
            (1 | 2, 2) => (level + phi * trend) * s,
            (1 | 2, _) => level + phi * trend,
            (_, 1) => level * trend + s,
            (_, 2) => level * trend * s,
            (_, _) => level * trend,
        };

        fitted[t] = yhat;
        let error = y[t] - yhat;
        residuals[t] = error;

        let level_old = level;
        let trend_old = trend;

        if error_type == 0 {
            level = match seasonal_type {
                2 => (level_old + phi * trend_old) + alpha * error / (s + 1e-10),
                _ => level_old + phi * trend_old + alpha * error,
            };
        } else {
            let e_ratio = error / (yhat + 1e-10);
            level = (level_old + phi * trend_old) * (1.0 + alpha * e_ratio);
        }

        level = level.clamp(-1e15, 1e15);

        match trend_type {
            1 | 2 => {
                trend = phi * trend_old + beta * (level - level_old);
            }
            3 => {
                trend = trend_old * (level / (level_old + 1e-10)).powf(beta);
            }
            _ => {}
        }

        match seasonal_type {
            1 => {
                seasonal[season_idx] = s + gamma * error;
            }
            2 => {
                if error_type == 0 {
                    seasonal[season_idx] =
                        s + gamma * error / (level_old + phi * trend_old + 1e-10);
                } else {
                    seasonal[season_idx] = s * (1.0 + gamma * error / (yhat + 1e-10));
                }
                if seasonal[season_idx] < 1e-6 {
                    seasonal[season_idx] = 1e-6;
                }
            }
            _ => {}
        }
    }

    let seasonal_out = Array1::from_vec(seasonal);
    Ok((
        fitted.into_pyarray(py).into(),
        residuals.into_pyarray(py).into(),
        level,
        trend,
        seasonal_out.into_pyarray(py).into(),
    ))
}

#[pyfunction]
fn ets_loglik(
    y: PyReadonlyArray1<f64>,
    level0: f64,
    trend0: f64,
    seasonal0: PyReadonlyArray1<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    phi: f64,
    period: usize,
    error_type: i32,
    trend_type: i32,
    seasonal_type: i32,
) -> PyResult<f64> {
    let y = y.as_array();
    let n = y.len();
    let mut seasonal: Vec<f64> = seasonal0.as_array().to_vec();

    let mut level = level0;
    let mut trend = trend0;
    let mut sse = 0.0f64;
    let mut sum_log_y = 0.0f64;

    for t in 0..n {
        let season_idx = t % period;

        let s = match seasonal_type {
            1 => seasonal[season_idx],
            2 => {
                if seasonal[season_idx] < 1e-6 {
                    seasonal[season_idx] = 1e-6;
                }
                seasonal[season_idx]
            }
            _ => 0.0,
        };

        let yhat = match (trend_type, seasonal_type) {
            (0, 1) => level + s,
            (0, 2) => level * s,
            (0, _) => level,
            (1 | 2, 1) => level + phi * trend + s,
            (1 | 2, 2) => (level + phi * trend) * s,
            (1 | 2, _) => level + phi * trend,
            (_, 1) => level * trend + s,
            (_, 2) => level * trend * s,
            (_, _) => level * trend,
        };

        let error = y[t] - yhat;

        if error_type == 0 {
            sse += error * error;
        } else {
            let e_ratio = error / (yhat.abs() + 1e-10);
            sse += e_ratio * e_ratio;
            sum_log_y += yhat.abs().max(1e-10).ln();
        }

        let level_old = level;
        let trend_old = trend;

        if error_type == 0 {
            level = match seasonal_type {
                2 => (level_old + phi * trend_old) + alpha * error / (s + 1e-10),
                _ => level_old + phi * trend_old + alpha * error,
            };
        } else {
            let e_ratio = error / (yhat + 1e-10);
            level = (level_old + phi * trend_old) * (1.0 + alpha * e_ratio);
        }

        level = level.clamp(-1e15, 1e15);

        match trend_type {
            1 | 2 => {
                trend = phi * trend_old + beta * (level - level_old);
            }
            3 => {
                trend = trend_old * (level / (level_old + 1e-10)).powf(beta);
            }
            _ => {}
        }

        match seasonal_type {
            1 => {
                seasonal[season_idx] = s + gamma * error;
            }
            2 => {
                if error_type == 0 {
                    seasonal[season_idx] =
                        s + gamma * error / (level_old + phi * trend_old + 1e-10);
                } else {
                    seasonal[season_idx] = s * (1.0 + gamma * error / (yhat + 1e-10));
                }
                if seasonal[season_idx] < 1e-6 {
                    seasonal[season_idx] = 1e-6;
                }
            }
            _ => {}
        }
    }

    let nf = n as f64;
    let loglik = if error_type == 0 {
        nf * (1.0 + (2.0 * std::f64::consts::PI * sse / nf).ln())
    } else {
        nf * (1.0 + (2.0 * std::f64::consts::PI * sse / nf).ln()) + 2.0 * sum_log_y
    };

    Ok(loglik)
}

#[pyfunction]
fn theta_decompose(py: Python<'_>, y: PyReadonlyArray1<f64>, theta: f64) -> PyResult<Py<PyArray1<f64>>> {
    let y = y.as_array();
    let n = y.len();
    let mut result = Array1::<f64>::zeros(n);

    let nf = n as f64;
    for i in 0..n {
        let t = (i as f64 + 1.0) / nf;
        result[i] = theta * y[i] + (1.0 - theta) * (y[0] + (y[n - 1] - y[0]) * t);
    }

    Ok(result.into_pyarray(py).into())
}

#[pyfunction]
fn arima_css(
    y: PyReadonlyArray1<f64>,
    ar: PyReadonlyArray1<f64>,
    ma: PyReadonlyArray1<f64>,
    d: usize,
) -> PyResult<f64> {
    let y = y.as_array();
    let ar = ar.as_array();
    let ma = ma.as_array();

    let mut diff = y.to_vec();
    for _ in 0..d {
        let prev = diff.clone();
        diff = prev.windows(2).map(|w| w[1] - w[0]).collect();
    }

    let n = diff.len();
    let p = ar.len();
    let q = ma.len();
    let start = p.max(q);

    if n <= start {
        return Ok(f64::MAX);
    }

    let mut residuals = vec![0.0f64; n];

    for t in start..n {
        let mut pred = 0.0f64;
        for j in 0..p {
            pred += ar[j] * diff[t - 1 - j];
        }
        for j in 0..q.min(t) {
            pred += ma[j] * residuals[t - 1 - j];
        }
        residuals[t] = diff[t] - pred;
    }

    let valid = &residuals[start..];
    let sse: f64 = valid.iter().map(|r| r * r).sum();
    let nv = valid.len() as f64;

    Ok(nv * (sse / nv).ln())
}

#[pyfunction]
fn css_objective(
    y: PyReadonlyArray1<f64>,
    ar_coefs: PyReadonlyArray1<f64>,
    ma_coefs: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let y = y.as_array();
    let ar = ar_coefs.as_array();
    let ma = ma_coefs.as_array();
    let n = y.len();
    let p = ar.len();
    let q = ma.len();
    let max_lag = p.max(q).max(1);

    let mut residuals = vec![0.0f64; n];
    let mut css = 0.0f64;

    for t in max_lag..n {
        let mut pred = 0.0f64;
        for i in 0..p {
            pred += ar[i] * y[t - i - 1];
        }
        for j in 0..q.min(t) {
            pred += ma[j] * residuals[t - j - 1];
        }
        residuals[t] = y[t] - pred;
        css += residuals[t] * residuals[t];
    }

    Ok(css)
}

#[pyfunction]
fn seasonal_css_objective(
    y: PyReadonlyArray1<f64>,
    ar_coefs: PyReadonlyArray1<f64>,
    ma_coefs: PyReadonlyArray1<f64>,
    sar_coefs: PyReadonlyArray1<f64>,
    sma_coefs: PyReadonlyArray1<f64>,
    m: usize,
) -> PyResult<f64> {
    let y = y.as_array();
    let ar = ar_coefs.as_array();
    let ma = ma_coefs.as_array();
    let sar = sar_coefs.as_array();
    let sma = sma_coefs.as_array();
    let n = y.len();
    let p = ar.len();
    let q = ma.len();
    let big_p = sar.len();
    let big_q = sma.len();
    let max_lag = p.max(q).max(big_p * m).max(big_q * m).max(1);

    let mut residuals = vec![0.0f64; n];
    let mut css = 0.0f64;

    for t in max_lag..n {
        let mut pred = 0.0f64;
        for i in 0..p {
            pred += ar[i] * y[t - i - 1];
        }
        for i in 0..big_p {
            let idx = t as isize - ((i + 1) * m) as isize;
            if idx >= 0 {
                pred += sar[i] * y[idx as usize];
            }
        }
        for j in 0..q.min(t) {
            pred += ma[j] * residuals[t - j - 1];
        }
        for j in 0..big_q {
            let idx = t as isize - ((j + 1) * m) as isize;
            if idx >= 0 {
                pred += sma[j] * residuals[idx as usize];
            }
        }
        residuals[t] = y[t] - pred;
        css += residuals[t] * residuals[t];
    }

    Ok(css)
}

#[pyfunction]
fn ses_sse(y: PyReadonlyArray1<f64>, alpha: f64) -> PyResult<f64> {
    let y = y.as_array();
    let n = y.len();
    let mut level = y[0];
    let mut sse = 0.0f64;

    for t in 1..n {
        let error = y[t] - level;
        sse += error * error;
        level = alpha * y[t] + (1.0 - alpha) * level;
    }

    Ok(sse)
}

#[pyfunction]
fn ses_filter(py: Python<'_>, y: PyReadonlyArray1<f64>, alpha: f64) -> PyResult<Py<PyArray1<f64>>> {
    let y = y.as_array();
    let n = y.len();
    let mut result = Array1::<f64>::zeros(n);
    result[0] = y[0];

    for t in 1..n {
        result[t] = alpha * y[t] + (1.0 - alpha) * result[t - 1];
    }

    Ok(result.into_pyarray(py).into())
}

#[pyfunction]
fn batch_ets_filter(
    py: Python<'_>,
    y_list: Vec<PyReadonlyArray1<f64>>,
    params_list: Vec<(f64, f64, Vec<f64>, f64, f64, f64, f64, usize, i32, i32, i32)>,
) -> PyResult<Vec<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, f64, f64, Py<PyArray1<f64>>)>> {
    let mut results = Vec::with_capacity(y_list.len());

    for (y_ro, params) in y_list.iter().zip(params_list.iter()) {
        let y = y_ro.as_array();
        let n = y.len();
        let (level0, trend0, ref seasonal0, alpha, beta, gamma, phi, period, error_type, trend_type, seasonal_type) = *params;

        let mut fitted = Array1::<f64>::zeros(n);
        let mut residuals = Array1::<f64>::zeros(n);
        let mut seasonal = seasonal0.clone();
        let mut level = level0;
        let mut trend = trend0;

        for t in 0..n {
            let season_idx = t % period;
            let s = match seasonal_type {
                1 => seasonal[season_idx],
                2 => {
                    if seasonal[season_idx] < 1e-6 { seasonal[season_idx] = 1e-6; }
                    seasonal[season_idx]
                }
                _ => 0.0,
            };

            let yhat = match (trend_type, seasonal_type) {
                (0, 1) => level + s,
                (0, 2) => level * s,
                (0, _) => level,
                (1 | 2, 1) => level + phi * trend + s,
                (1 | 2, 2) => (level + phi * trend) * s,
                (1 | 2, _) => level + phi * trend,
                (_, 1) => level * trend + s,
                (_, 2) => level * trend * s,
                (_, _) => level * trend,
            };

            fitted[t] = yhat;
            let error = y[t] - yhat;
            residuals[t] = error;
            let level_old = level;
            let trend_old = trend;

            if error_type == 0 {
                level = match seasonal_type {
                    2 => (level_old + phi * trend_old) + alpha * error / (s + 1e-10),
                    _ => level_old + phi * trend_old + alpha * error,
                };
            } else {
                let e_ratio = error / (yhat + 1e-10);
                level = (level_old + phi * trend_old) * (1.0 + alpha * e_ratio);
            }
            level = level.clamp(-1e15, 1e15);

            match trend_type {
                1 | 2 => { trend = phi * trend_old + beta * (level - level_old); }
                3 => { trend = trend_old * (level / (level_old + 1e-10)).powf(beta); }
                _ => {}
            }

            match seasonal_type {
                1 => { seasonal[season_idx] = s + gamma * error; }
                2 => {
                    if error_type == 0 {
                        seasonal[season_idx] = s + gamma * error / (level_old + phi * trend_old + 1e-10);
                    } else {
                        seasonal[season_idx] = s * (1.0 + gamma * error / (yhat + 1e-10));
                    }
                    if seasonal[season_idx] < 1e-6 { seasonal[season_idx] = 1e-6; }
                }
                _ => {}
            }
        }

        let seasonal_out = Array1::from_vec(seasonal);
        results.push((
            fitted.into_pyarray(py).into(),
            residuals.into_pyarray(py).into(),
            level,
            trend,
            seasonal_out.into_pyarray(py).into(),
        ));
    }

    Ok(results)
}

#[pyfunction]
fn dot_objective(
    y: PyReadonlyArray1<f64>,
    intercept: f64,
    slope: f64,
    theta: f64,
    alpha: f64,
    drift: f64,
) -> PyResult<f64> {
    let y = y.as_array();
    let n = y.len();
    if n < 2 {
        return Ok(0.0);
    }

    let mut theta_line = vec![0.0f64; n];
    for i in 0..n {
        let linear = intercept + slope * (i as f64);
        theta_line[i] = theta * y[i] + (1.0 - theta) * linear;
    }

    let mut level = theta_line[0];
    let mut sse = 0.0f64;
    for t in 1..n {
        let trend_pred = intercept + slope * (t as f64);
        let pred = (trend_pred + level + drift * (t as f64)) / 2.0;
        let error = y[t] - pred;
        sse += error * error;
        level = alpha * theta_line[t] + (1.0 - alpha) * level;
    }

    Ok(sse)
}

#[pyfunction]
fn dot_residuals(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    intercept: f64,
    slope: f64,
    theta: f64,
    alpha: f64,
    drift: f64,
) -> PyResult<(Py<PyArray1<f64>>, f64)> {
    let y = y.as_array();
    let n = y.len();
    if n < 2 {
        return Ok((Array1::<f64>::zeros(0).into_pyarray(py).into(), 0.0));
    }

    let mut theta_line = vec![0.0f64; n];
    for i in 0..n {
        let linear = intercept + slope * (i as f64);
        theta_line[i] = theta * y[i] + (1.0 - theta) * linear;
    }

    let mut level = theta_line[0];
    let mut residuals = Array1::<f64>::zeros(n - 1);
    for t in 1..n {
        let trend_pred = intercept + slope * (t as f64);
        let pred = (trend_pred + level + drift * (t as f64)) / 2.0;
        residuals[t - 1] = y[t] - pred;
        level = alpha * theta_line[t] + (1.0 - alpha) * level;
    }

    Ok((residuals.into_pyarray(py).into(), level))
}

#[pyfunction]
fn ces_nonseasonal_sse(
    y: PyReadonlyArray1<f64>,
    a0_real: f64,
    a0_imag: f64,
) -> PyResult<f64> {
    let y = y.as_array();
    let n = y.len();
    if n < 2 {
        return Ok(0.0);
    }

    let mut level_real = y[0];
    let mut level_imag = 0.0f64;
    let mut sse = 0.0f64;

    for t in 1..n {
        let forecast = level_real;
        let error = y[t] - forecast;
        sse += error * error;
        let new_real = a0_real * y[t] + (1.0 - a0_real) * level_real + a0_imag * level_imag;
        let new_imag = a0_imag * (y[t] - level_real) + (1.0 - a0_real) * level_imag;
        level_real = new_real;
        level_imag = new_imag;
    }

    Ok(sse)
}

#[pyfunction]
fn ces_seasonal_sse(
    y: PyReadonlyArray1<f64>,
    a0_real: f64,
    a0_imag: f64,
    gamma: f64,
    seasonal_init: PyReadonlyArray1<f64>,
    m: usize,
) -> PyResult<f64> {
    let y = y.as_array();
    let n = y.len();
    if n < 2 || m == 0 {
        return Ok(0.0);
    }

    let mut seasonal: Vec<f64> = seasonal_init.as_array().to_vec();
    let mut level_real = y[0] - seasonal[0];
    let mut level_imag = 0.0f64;
    let mut sse = 0.0f64;

    for t in 1..n {
        let sidx = t % m;
        let forecast = level_real + seasonal[sidx];
        let error = y[t] - forecast;
        sse += error * error;
        let y_adj = y[t] - seasonal[sidx];
        let new_real = a0_real * y_adj + (1.0 - a0_real) * level_real + a0_imag * level_imag;
        let new_imag = a0_imag * (y_adj - level_real) + (1.0 - a0_real) * level_imag;
        level_real = new_real;
        level_imag = new_imag;
        seasonal[sidx] += gamma * error;
    }

    Ok(sse)
}

#[pymodule]
fn vectrix_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ets_filter, m)?)?;
    m.add_function(wrap_pyfunction!(ets_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(theta_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(arima_css, m)?)?;
    m.add_function(wrap_pyfunction!(css_objective, m)?)?;
    m.add_function(wrap_pyfunction!(seasonal_css_objective, m)?)?;
    m.add_function(wrap_pyfunction!(ses_sse, m)?)?;
    m.add_function(wrap_pyfunction!(ses_filter, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ets_filter, m)?)?;
    m.add_function(wrap_pyfunction!(dot_objective, m)?)?;
    m.add_function(wrap_pyfunction!(dot_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(ces_nonseasonal_sse, m)?)?;
    m.add_function(wrap_pyfunction!(ces_seasonal_sse, m)?)?;
    Ok(())
}
