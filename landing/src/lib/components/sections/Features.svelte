<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { Zap, Clock, Shield, Wrench, BookOpen, Briefcase, CheckCircle, BarChart3, Activity } from 'lucide-svelte';

	const features = [
		{ icon: Zap, color: 'text-vx-primary bg-vx-primary/10', title: 'Zero Configuration', desc: 'Pass your data, get forecasts. Auto model selection, hyperparameter tuning, and flat-line defense — all automatic.' },
		{ icon: Clock, color: 'text-vx-accent bg-vx-accent/10', title: '30+ Statistical Models', desc: 'ETS, ARIMA, Theta, MSTL, TBATS, GARCH, CES, Croston, DOT, 4Theta, ESN, DTSF — and more.' },
		{ icon: Shield, color: 'text-vx-success bg-vx-success/10', title: 'Flat Defense System', desc: 'Unique 4-level system prevents constant predictions — the most common forecasting failure mode in production.' },
		{ icon: Wrench, color: 'text-vx-warning bg-vx-warning/10', title: 'Forecast DNA', desc: '65+ feature fingerprinting. Know your data\'s difficulty, optimal model, seasonality, and similarity to other series.' },
		{ icon: BookOpen, color: 'text-vx-primary bg-vx-primary/10', title: 'Adaptive Intelligence', desc: 'Regime detection (HMM), self-healing forecasts (CUSUM + EWMA), and 8 business constraint types.' },
		{ icon: Briefcase, color: 'text-vx-accent bg-vx-accent/10', title: 'Pure Python', desc: 'Built on NumPy + SciPy + Pandas only. No compiled extensions, no platform issues. Optional Rust turbo for 24x speed.' },
		{ icon: CheckCircle, color: 'text-vx-success bg-vx-success/10', title: 'Probabilistic Forecasting', desc: 'Parametric distributions (Gaussian, Student-t, Log-Normal), quantile forecasts, conformal prediction intervals, CRPS.' },
		{ icon: BarChart3, color: 'text-vx-warning bg-vx-warning/10', title: 'R-Style Regression', desc: 'Formula interface: regress(df, "sales ~ ads + price"). OLS, Ridge, Lasso, Huber, Quantile with full diagnostics.' },
		{ icon: Activity, color: 'text-vx-primary bg-vx-primary/10', title: 'Business Intelligence', desc: 'Anomaly detection, what-if scenarios, backtesting framework, and production metrics (WAPE, MASE, bias).' }
	];
</script>

<section class="py-20 px-6 max-w-6xl mx-auto text-center">
	<span class="text-xs font-semibold uppercase tracking-widest text-vx-primary mb-3 block">Features</span>
	<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-3">Everything you need for forecasting</h2>
	<p class="text-vx-text-muted max-w-xl mx-auto mb-12">
		From one-line forecasts to advanced adaptive intelligence — Vectrix scales with your needs.
	</p>

	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
		{#each features as f}
			<Card class="text-left">
				<div class="w-11 h-11 rounded-lg flex items-center justify-center mb-4 {f.color}">
					<f.icon class="w-5 h-5" />
				</div>
				<h3 class="text-base font-semibold text-vx-text mb-2">{f.title}</h3>
				<p class="text-sm text-vx-text-muted leading-relaxed">{f.desc}</p>
			</Card>
		{/each}
	</div>
</section>
