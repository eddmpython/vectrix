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

// ── GARCH(1,1) negative log-likelihood ──────────────────────────────────────
#[pyfunction]
fn garch_filter(
    y: PyReadonlyArray1<f64>,
    mu: f64,
    omega: f64,
    alpha_g: f64,
    beta_g: f64,
    init_sigma2: f64,
    use_ar1: bool,
    ar1: f64,
) -> PyResult<f64> {
    let y = y.as_array();
    let n = y.len();
    let mut sigma2 = init_sigma2;
    let mut nll = 0.0f64;

    for t in 0..n {
        let mean_pred = if use_ar1 && t > 0 {
            mu + ar1 * (y[t - 1] - mu)
        } else {
            mu
        };
        let eps = y[t] - mean_pred;
        let s2 = sigma2.max(1e-20);
        nll += 0.5 * (s2.ln() + eps * eps / s2);
        sigma2 = omega + alpha_g * eps * eps + beta_g * sigma2;
    }

    Ok(nll)
}

// ── EGARCH(1,1) negative log-likelihood ─────────────────────────────────────
#[pyfunction]
fn egarch_filter(
    y: PyReadonlyArray1<f64>,
    mu: f64,
    omega: f64,
    alpha_e: f64,
    beta_e: f64,
    gamma: f64,
    init_log_sigma2: f64,
) -> PyResult<f64> {
    let y = y.as_array();
    let n = y.len();
    let sqrt2pi = (2.0f64 / std::f64::consts::PI).sqrt();
    let mut log_sigma2 = init_log_sigma2;
    let mut nll = 0.0f64;

    for t in 0..n {
        let sigma2 = log_sigma2.exp();
        let sigma = sigma2.max(1e-20).sqrt();
        let eps = y[t] - mu;
        let z = eps / sigma.max(1e-10);

        nll += 0.5 * (log_sigma2 + eps * eps / sigma2.max(1e-20));

        let g = alpha_e * z + gamma * (z.abs() - sqrt2pi);
        log_sigma2 = omega + g + beta_e * log_sigma2;
        log_sigma2 = log_sigma2.clamp(-20.0, 20.0);
    }

    Ok(nll)
}

// ── GJR-GARCH(1,1) negative log-likelihood ──────────────────────────────────
#[pyfunction]
fn gjr_garch_filter(
    y: PyReadonlyArray1<f64>,
    mu: f64,
    omega: f64,
    alpha_gjr: f64,
    beta_gjr: f64,
    gamma_gjr: f64,
    init_sigma2: f64,
) -> PyResult<f64> {
    let y = y.as_array();
    let n = y.len();
    let mut sigma2 = init_sigma2;
    let mut nll = 0.0f64;

    for t in 0..n {
        let eps = y[t] - mu;
        let s2 = sigma2.max(1e-20);
        nll += 0.5 * (s2.ln() + eps * eps / s2);
        let indicator = if eps < 0.0 { 1.0 } else { 0.0 };
        sigma2 = omega + (alpha_gjr + gamma_gjr * indicator) * eps * eps + beta_gjr * sigma2;
    }

    Ok(nll)
}

// ── TBATS filter (Fourier state update loop) ────────────────────────────────
#[pyfunction]
fn tbats_filter(
    y: PyReadonlyArray1<f64>,
    alpha: f64,
    beta: f64,
    phi: f64,
    use_trend: bool,
    use_damping: bool,
    frequencies: PyReadonlyArray1<f64>,
    total_harmonics: usize,
) -> PyResult<f64> {
    let y = y.as_array();
    let n = y.len();
    let frequencies = frequencies.as_array();
    let gamma = alpha * 0.5;

    let mut level = y[0];
    let mut trend = 0.0f64;
    let mut sj = vec![0.0f64; total_harmonics];
    let mut sj_star = vec![0.0f64; total_harmonics];
    let mut sse = 0.0f64;

    for t in 1..n {
        let mut pred = level;
        if use_trend {
            pred += trend;
        }
        for h in 0..total_harmonics {
            pred += sj[h];
        }

        let error = y[t] - pred;
        sse += error * error;

        level += alpha * error;
        if use_trend {
            if use_damping {
                trend = phi * trend + beta * error;
            } else {
                trend += beta * error;
            }
        }

        for h in 0..total_harmonics {
            let cos_f = frequencies[h].cos();
            let sin_f = frequencies[h].sin();
            let new_sj = sj[h] * cos_f + sj_star[h] * sin_f + gamma * error;
            let new_sj_star = -sj[h] * sin_f + sj_star[h] * cos_f;
            sj[h] = new_sj;
            sj_star[h] = new_sj_star;
        }
    }

    Ok(sse)
}

// ── DTSF sliding window distance computation ────────────────────────────────
#[pyfunction]
fn dtsf_distances(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    query_norm: PyReadonlyArray1<f64>,
    w: usize,
    max_start: usize,
    normalize: bool,
    time_decay: f64,
    n_total: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let y = y.as_array();
    let query = query_norm.as_array();
    let w_f = w as f64;
    let mut distances = Array1::<f64>::zeros(max_start);

    for i in 0..max_start {
        let window = &y.as_slice().unwrap()[i..i + w];

        let shape_dist = if normalize {
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for &v in window {
                sum += v;
                sum_sq += v * v;
            }
            let w_mean = sum / w_f;
            let w_std = ((sum_sq / w_f - w_mean * w_mean).max(0.0)).sqrt().max(1e-8);

            let mut dist_sum = 0.0f64;
            for j in 0..w {
                let norm_val = (window[j] - w_mean) / w_std;
                let diff = query[j] - norm_val;
                dist_sum += diff * diff;
            }
            (dist_sum / w_f).sqrt()
        } else {
            let mut dist_sum = 0.0f64;
            for j in 0..w {
                let diff = query[j] - window[j];
                dist_sum += diff * diff;
            }
            (dist_sum / w_f).sqrt()
        };

        let time_weight = (-time_decay * (n_total - i) as f64).exp();
        distances[i] = shape_dist / time_weight.max(1e-10);
    }

    Ok(distances.into_pyarray(py).into())
}

// ── DTSF fit residuals (O(n^2) double loop) ─────────────────────────────────
#[pyfunction]
fn dtsf_fit_residuals(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    w: usize,
    n_neighbors: usize,
    normalize: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let y = y.as_array();
    let n = y.len();
    let w_f = w as f64;
    let mut residuals = Array1::<f64>::zeros(n);

    for t in w..n {
        let query = &y.as_slice().unwrap()[t - w..t];

        let (q_mean, q_std) = if normalize {
            let mut s = 0.0f64;
            let mut s2 = 0.0f64;
            for &v in query {
                s += v;
                s2 += v * v;
            }
            let m = s / w_f;
            let std = ((s2 / w_f - m * m).max(0.0)).sqrt().max(1e-8);
            (m, std)
        } else {
            (0.0, 1.0)
        };

        let max_start = t - w;
        if max_start < 1 {
            continue;
        }

        let mut distances = vec![0.0f64; max_start];
        for i in 0..max_start {
            let window = &y.as_slice().unwrap()[i..i + w];
            if normalize {
                let mut s = 0.0f64;
                let mut s2 = 0.0f64;
                for &v in window {
                    s += v;
                    s2 += v * v;
                }
                let w_mean = s / w_f;
                let w_std = ((s2 / w_f - w_mean * w_mean).max(0.0)).sqrt().max(1e-8);

                let mut dist_sum = 0.0f64;
                for j in 0..w {
                    let qn = (query[j] - q_mean) / q_std;
                    let wn = (window[j] - w_mean) / w_std;
                    let diff = qn - wn;
                    dist_sum += diff * diff;
                }
                distances[i] = (dist_sum / w_f).sqrt();
            } else {
                let mut dist_sum = 0.0f64;
                for j in 0..w {
                    let diff = query[j] - window[j];
                    dist_sum += diff * diff;
                }
                distances[i] = (dist_sum / w_f).sqrt();
            }
        }

        let k = n_neighbors.min(max_start);
        let mut indices: Vec<usize> = (0..max_start).collect();
        indices.select_nth_unstable_by(k.min(max_start) - 1, |&a, &b| {
            distances[a].partial_cmp(&distances[b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut preds = Vec::with_capacity(k);
        for &idx in &indices[..k] {
            let next_idx = idx + w;
            if next_idx < n {
                preds.push(y[next_idx]);
            }
        }

        if !preds.is_empty() {
            preds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if preds.len() % 2 == 0 {
                (preds[preds.len() / 2 - 1] + preds[preds.len() / 2]) / 2.0
            } else {
                preds[preds.len() / 2]
            };
            residuals[t] = y[t] - median;
        }
    }

    Ok(residuals.into_pyarray(py).into())
}

// ── MSTL extract seasonal + moving average ──────────────────────────────────
#[pyfunction]
fn mstl_extract_seasonal(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let y = y.as_array();
    let n = y.len();
    let mut period_sums = vec![0.0f64; period];
    let mut period_counts = vec![0usize; period];

    for i in 0..n {
        period_sums[i % period] += y[i];
        period_counts[i % period] += 1;
    }

    let mut period_means = vec![0.0f64; period];
    for i in 0..period {
        period_means[i] = period_sums[i] / period_counts[i].max(1) as f64;
    }

    let global_mean: f64 = period_means.iter().sum::<f64>() / period as f64;
    for v in &mut period_means {
        *v -= global_mean;
    }

    let mut seasonal = Array1::<f64>::zeros(n);
    for i in 0..n {
        seasonal[i] = period_means[i % period];
    }

    Ok(seasonal.into_pyarray(py).into())
}

#[pyfunction]
fn mstl_moving_average(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let y = y.as_array();
    let n = y.len();
    let half_win = window / 2;
    let mut result = Array1::<f64>::zeros(n);

    let mut cumsum = vec![0.0f64; n + 1];
    for i in 0..n {
        cumsum[i + 1] = cumsum[i] + y[i];
    }

    for i in 0..n {
        let start = if i >= half_win { i - half_win } else { 0 };
        let end = (i + half_win + 1).min(n);
        result[i] = (cumsum[end] - cumsum[start]) / (end - start) as f64;
    }

    Ok(result.into_pyarray(py).into())
}

// ── Croston TSB filter ──────────────────────────────────────────────────────
#[pyfunction]
fn croston_tsb_filter(
    y: PyReadonlyArray1<f64>,
    alpha: f64,
    beta: f64,
    init_z: f64,
    init_d: f64,
) -> PyResult<(f64, f64, f64)> {
    let y = y.as_array();
    let n = y.len();
    let mut z = init_z;
    let mut d = init_d;
    let mut sse = 0.0f64;

    for t in 0..n {
        let forecast = d * z;
        let error = y[t] - forecast;
        sse += error * error;

        if y[t] > 0.0 {
            z = alpha * y[t] + (1.0 - alpha) * z;
            d = beta + (1.0 - beta) * d;
        } else {
            d = (1.0 - beta) * d;
        }
    }

    Ok((z, d, sse))
}

// ── ESN reservoir update loop ───────────────────────────────────────────────
#[pyfunction]
fn esn_reservoir_update(
    py: Python<'_>,
    y_norm: PyReadonlyArray1<f64>,
    w_in_flat: PyReadonlyArray1<f64>,
    w_flat: PyReadonlyArray1<f64>,
    reservoir_size: usize,
    leak_rate: f64,
    washout: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let y = y_norm.as_array();
    let w_in = w_in_flat.as_array();
    let w = w_flat.as_array();
    let n = y.len();
    let nn = reservoir_size;

    let states_rows = if n > washout { n - washout } else { 0 };
    let mut states = vec![0.0f64; states_rows * nn];
    let mut x = vec![0.0f64; nn];

    for t in 0..n {
        let u = y[t];
        let mut x_new = vec![0.0f64; nn];
        for i in 0..nn {
            let mut val = w_in[i] * u;
            for j in 0..nn {
                val += w[i * nn + j] * x[j];
            }
            x_new[i] = val.tanh();
        }
        for i in 0..nn {
            x[i] = (1.0 - leak_rate) * x[i] + leak_rate * x_new[i];
        }
        if t >= washout {
            let row = t - washout;
            for i in 0..nn {
                states[row * nn + i] = x[i];
            }
        }
    }

    let states_arr = Array1::from_vec(states);
    let x_arr = Array1::from_vec(x);

    Ok((
        states_arr.into_pyarray(py).into(),
        x_arr.into_pyarray(py).into(),
    ))
}

// ── FourTheta fitted values combination ─────────────────────────────────────
#[pyfunction]
fn four_theta_fitted(
    py: Python<'_>,
    n: usize,
    thetas: PyReadonlyArray1<f64>,
    intercepts: PyReadonlyArray1<f64>,
    slopes: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    filtered_flat: PyReadonlyArray1<f64>,
    filtered_lengths: PyReadonlyArray1<i64>,
    last_levels: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let thetas = thetas.as_array();
    let intercepts = intercepts.as_array();
    let slopes = slopes.as_array();
    let weights = weights.as_array();
    let filtered_flat = filtered_flat.as_array();
    let filtered_lengths = filtered_lengths.as_array();
    let last_levels = last_levels.as_array();
    let n_models = thetas.len();

    let mut offsets = vec![0usize; n_models];
    for i in 1..n_models {
        offsets[i] = offsets[i - 1] + filtered_lengths[i - 1] as usize;
    }

    let mut fitted = Array1::<f64>::zeros(n);

    for i in 0..n_models {
        let w = weights[i];
        let theta = thetas[i];
        let intercept = intercepts[i];
        let slope = slopes[i];
        let off = offsets[i];
        let flen = filtered_lengths[i] as usize;

        for t in 0..n {
            let trend_pred = intercept + slope * t as f64;
            let ses_pred = if t < flen {
                filtered_flat[off + t]
            } else {
                last_levels[i]
            };

            if theta == 0.0 {
                fitted[t] += w * trend_pred;
            } else {
                fitted[t] += w * (trend_pred + ses_pred) / 2.0;
            }
        }
    }

    Ok(fitted.into_pyarray(py).into())
}

// ── FourTheta deseasonalize ─────────────────────────────────────────────────
#[pyfunction]
fn four_theta_deseasonalize(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<(Py<PyArray1<f64>>, bool, Py<PyArray1<f64>>)> {
    let y = y.as_array();
    let n = y.len();
    let kernel: Vec<f64> = vec![1.0 / period as f64; period];
    let trend_len = n - period + 1;
    let offset = (period - 1) / 2;

    let min_val = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let use_mult = min_val > 0.0;

    let mut trend = vec![0.0f64; trend_len];
    for i in 0..trend_len {
        let mut s = 0.0f64;
        for j in 0..period {
            s += y[i + j] * kernel[j];
        }
        trend[i] = s;
    }

    let mut seasonal_sum = vec![0.0f64; period];
    let mut seasonal_count = vec![0usize; period];

    if use_mult {
        for i in 0..trend_len {
            let idx = i + offset;
            if idx < n && trend[i] > 0.0 {
                seasonal_sum[idx % period] += y[idx] / trend[i];
                seasonal_count[idx % period] += 1;
            }
        }
        let mut seasonal = vec![0.0f64; period];
        for i in 0..period {
            seasonal[i] = seasonal_sum[i] / seasonal_count[i].max(1) as f64;
        }
        let mean_s: f64 = seasonal.iter().sum::<f64>() / period as f64;
        if mean_s > 0.0 {
            for v in &mut seasonal {
                *v /= mean_s;
            }
        }
        for v in &mut seasonal {
            if *v < 0.01 { *v = 0.01; }
        }
        let mut deseason = Array1::<f64>::zeros(n);
        for i in 0..n {
            deseason[i] = y[i] / seasonal[i % period];
        }
        let seasonal_arr = Array1::from_vec(seasonal);
        Ok((seasonal_arr.into_pyarray(py).into(), true, deseason.into_pyarray(py).into()))
    } else {
        for i in 0..trend_len {
            let idx = i + offset;
            if idx < n {
                seasonal_sum[idx % period] += y[idx] - trend[i];
                seasonal_count[idx % period] += 1;
            }
        }
        let mut seasonal = vec![0.0f64; period];
        for i in 0..period {
            seasonal[i] = seasonal_sum[i] / seasonal_count[i].max(1) as f64;
        }
        let mean_s: f64 = seasonal.iter().sum::<f64>() / period as f64;
        for v in &mut seasonal {
            *v -= mean_s;
        }
        let mut deseason = Array1::<f64>::zeros(n);
        for i in 0..n {
            deseason[i] = y[i] - seasonal[i % period];
        }
        let seasonal_arr = Array1::from_vec(seasonal);
        Ok((seasonal_arr.into_pyarray(py).into(), false, deseason.into_pyarray(py).into()))
    }
}

// ── DOT Hybrid objective (buildThetaLine + SES optimize + combine + MAE) ──
#[pyfunction]
fn dot_hybrid_objective(
    y: PyReadonlyArray1<f64>,
    theta_line0: PyReadonlyArray1<f64>,
    theta: f64,
    is_additive: bool,
) -> PyResult<f64> {
    let y = y.as_array();
    let t0 = theta_line0.as_array();
    let n = y.len();
    if n < 3 {
        return Ok(f64::MAX);
    }

    let mut theta_line = vec![0.0f64; n];
    if is_additive {
        for i in 0..n {
            theta_line[i] = theta * y[i] + (1.0 - theta) * t0[i];
        }
    } else {
        for i in 0..n {
            let yv = if y[i] > 1e-10 { y[i] } else { 1e-10 };
            let tv = if t0[i] > 1e-10 { t0[i] } else { 1e-10 };
            theta_line[i] = yv.powf(theta) * tv.powf(1.0 - theta);
        }
    }

    let alpha = optimize_alpha_bounded(&theta_line);

    let mut filtered = vec![0.0f64; n];
    filtered[0] = theta_line[0];
    for t in 1..n {
        filtered[t] = alpha * theta_line[t] + (1.0 - alpha) * filtered[t - 1];
    }

    let mut mae = 0.0f64;
    if is_additive {
        let w = 1.0 / theta.max(1.0);
        let w2 = 1.0 - w;
        for i in 0..n {
            let fitted = w * filtered[i] + w2 * t0[i];
            mae += (y[i] - fitted).abs();
        }
    } else {
        let inv = 1.0 / theta.max(1.0);
        let inv2 = 1.0 - inv;
        for i in 0..n {
            let fv = if filtered[i] > 1e-10 { filtered[i] } else { 1e-10 };
            let tv = if t0[i] > 1e-10 { t0[i] } else { 1e-10 };
            let fitted = fv.powf(inv) * tv.powf(inv2);
            mae += (y[i] - fitted).abs();
        }
    }

    Ok(mae / n as f64)
}

fn ses_sse_inner(y: &[f64], alpha: f64) -> f64 {
    let mut level = y[0];
    let mut sse = 0.0f64;
    for t in 1..y.len() {
        let error = y[t] - level;
        sse += error * error;
        level = alpha * y[t] + (1.0 - alpha) * level;
    }
    sse
}

fn optimize_alpha_bounded(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 3 {
        return 0.3;
    }

    let (mut lo, mut hi) = (0.001f64, 0.999f64);
    let gr = (5.0f64.sqrt() - 1.0) / 2.0;

    for _ in 0..50 {
        let x1 = hi - gr * (hi - lo);
        let x2 = lo + gr * (hi - lo);
        let f1 = ses_sse_inner(y, x1);
        let f2 = ses_sse_inner(y, x2);
        if f1 < f2 {
            hi = x2;
        } else {
            lo = x1;
        }
    }

    (lo + hi) / 2.0
}

#[pyfunction]
fn sample_entropy(
    _py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    m: usize,
    r: f64,
) -> f64 {
    let y = y.as_array();
    let n = y.len();
    if n < 10 || r < 1e-10 {
        return 0.0;
    }

    let count_matches = |m_val: usize| -> usize {
        let n_t = if n > m_val { n - m_val } else { return 0 };
        let mut count: usize = 0;
        for i in 0..n_t {
            for j in (i + 1)..n_t {
                let mut max_diff: f64 = 0.0;
                let mut within = true;
                for k in 0..m_val {
                    let d = (y[i + k] - y[j + k]).abs();
                    if d > r {
                        within = false;
                        break;
                    }
                    if d > max_diff {
                        max_diff = d;
                    }
                }
                if within {
                    count += 1;
                }
            }
        }
        count
    };

    let a = count_matches(m + 1);
    let b = count_matches(m);

    if b == 0 || a == 0 {
        return 0.0;
    }

    -(a as f64 / b as f64).ln()
}

#[pyfunction]
fn approximate_entropy(
    _py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    m: usize,
    r: f64,
) -> f64 {
    let y = y.as_array();
    let n = y.len();
    if n < 10 || r < 1e-10 {
        return 0.0;
    }

    let phi = |m_val: usize| -> f64 {
        let n_t = if n >= m_val { n - m_val + 1 } else { return 0.0 };
        let mut log_sum: f64 = 0.0;
        for i in 0..n_t {
            let mut count: usize = 0;
            for j in 0..n_t {
                let mut max_diff: f64 = 0.0;
                let mut within = true;
                for k in 0..m_val {
                    let d = (y[i + k] - y[j + k]).abs();
                    if d > r {
                        within = false;
                        break;
                    }
                    if d > max_diff {
                        max_diff = d;
                    }
                }
                if within {
                    count += 1;
                }
            }
            log_sum += ((count as f64) / (n_t as f64) + 1e-20).ln();
        }
        log_sum / n_t as f64
    };

    let result = phi(m) - phi(m + 1);
    if result > 0.0 { result } else { 0.0 }
}

#[pyfunction]
fn hurst_exponent(
    _py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    max_lag: usize,
) -> f64 {
    let y = y.as_array();
    let n = y.len();
    if n < 20 {
        return 0.5;
    }

    let mut log_lags: Vec<f64> = Vec::new();
    let mut log_rs: Vec<f64> = Vec::new();

    for lag in 2..=(max_lag.min(n / 4)) {
        let n_blocks = n / lag;
        if n_blocks < 1 {
            continue;
        }

        let mut rs_sum: f64 = 0.0;
        let mut rs_count: usize = 0;

        for b in 0..n_blocks {
            let start = b * lag;
            let end = start + lag;
            let block = &y.as_slice().unwrap()[start..end];

            let mean: f64 = block.iter().sum::<f64>() / lag as f64;

            let mut cum = 0.0_f64;
            let mut cum_min = f64::MAX;
            let mut cum_max = f64::MIN;
            let mut sq_sum = 0.0_f64;

            for &v in block {
                let d = v - mean;
                cum += d;
                if cum < cum_min { cum_min = cum; }
                if cum > cum_max { cum_max = cum; }
                sq_sum += d * d;
            }

            let r = cum_max - cum_min;
            let variance = sq_sum / (lag as f64 - 1.0);
            let s = variance.sqrt();

            if s > 1e-10 {
                rs_sum += r / s;
                rs_count += 1;
            }
        }

        if rs_count > 0 {
            let rs_mean = rs_sum / rs_count as f64;
            log_lags.push((lag as f64).ln());
            log_rs.push(rs_mean.ln());
        }
    }

    if log_lags.len() < 2 {
        return 0.5;
    }

    let n_pts = log_lags.len() as f64;
    let sum_x: f64 = log_lags.iter().sum();
    let sum_y: f64 = log_rs.iter().sum();
    let sum_xy: f64 = log_lags.iter().zip(log_rs.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = log_lags.iter().map(|x| x * x).sum();

    let denom = n_pts * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.5;
    }

    let slope = (n_pts * sum_xy - sum_x * sum_y) / denom;
    slope.max(0.0).min(1.0)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(garch_filter, m)?)?;
    m.add_function(wrap_pyfunction!(egarch_filter, m)?)?;
    m.add_function(wrap_pyfunction!(gjr_garch_filter, m)?)?;
    m.add_function(wrap_pyfunction!(tbats_filter, m)?)?;
    m.add_function(wrap_pyfunction!(dtsf_distances, m)?)?;
    m.add_function(wrap_pyfunction!(dtsf_fit_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(mstl_extract_seasonal, m)?)?;
    m.add_function(wrap_pyfunction!(mstl_moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(croston_tsb_filter, m)?)?;
    m.add_function(wrap_pyfunction!(esn_reservoir_update, m)?)?;
    m.add_function(wrap_pyfunction!(four_theta_fitted, m)?)?;
    m.add_function(wrap_pyfunction!(four_theta_deseasonalize, m)?)?;
    m.add_function(wrap_pyfunction!(dot_hybrid_objective, m)?)?;
    m.add_function(wrap_pyfunction!(sample_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(approximate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(hurst_exponent, m)?)?;
    Ok(())
}
