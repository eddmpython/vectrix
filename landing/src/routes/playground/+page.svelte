<script lang="ts">
	import { base } from '$app/paths';
	import Header from '$lib/components/sections/Header.svelte';
	import Footer from '$lib/components/sections/Footer.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Card } from '$lib/components/ui/card';
	import { Download, Play, BarChart3, Dna, Layers, Activity, TrendingUp, Zap, Clock, Database } from 'lucide-svelte';

	let { data } = $props();
	let datasets: any[] = data.playgroundData.datasets;
	let selectedId = $state('airline');
	let activeTab = $state<'forecast' | 'dna' | 'models'>('forecast');

	let selected = $derived(datasets.find((d: any) => d.id === selectedId));
	let ts = $derived(selected?.timeSeries);
	let fc = $derived(selected?.forecast);
	let dna = $derived(selected?.analysis?.dna);
	let chars = $derived(selected?.analysis?.characteristics);

	const domainIcons: Record<string, string> = {
		Transportation: '✈',
		Retail: '🛒',
		Finance: '📈',
		Energy: '⚡',
		Technology: '🌐',
		'Supply Chain': '📦'
	};

	const difficultyColors: Record<string, string> = {
		easy: 'text-vx-success',
		medium: 'text-vx-warning',
		hard: 'text-orange-400',
		very_hard: 'text-red-400'
	};

	function formatNum(v: number | null | undefined, decimals = 2): string {
		if (v == null || !isFinite(v)) return '—';
		return v.toFixed(decimals);
	}

	function svgPath(dates: string[], values: number[], width: number, height: number, padding = 0): string {
		if (!values?.length) return '';
		const minV = Math.min(...values.filter(v => v != null && isFinite(v)));
		const maxV = Math.max(...values.filter(v => v != null && isFinite(v)));
		const range = maxV - minV || 1;
		const xStep = (width - padding * 2) / (values.length - 1);
		return values
			.map((v, i) => {
				const x = padding + i * xStep;
				const y = height - padding - ((v - minV) / range) * (height - padding * 2);
				return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join(' ');
	}

	function svgArea(values: number[], lowers: number[], uppers: number[], width: number, height: number, padding = 0): string {
		if (!values?.length) return '';
		const allVals = [...values, ...lowers.filter(v => v != null), ...uppers.filter(v => v != null)];
		const minV = Math.min(...allVals.filter(v => isFinite(v)));
		const maxV = Math.max(...allVals.filter(v => isFinite(v)));
		const range = maxV - minV || 1;
		const xStep = (width - padding * 2) / (uppers.length - 1);

		const topPoints = uppers.map((v, i) => {
			const x = padding + i * xStep;
			const y = height - padding - ((v - minV) / range) * (height - padding * 2);
			return `${x.toFixed(1)},${y.toFixed(1)}`;
		});
		const bottomPoints = lowers.map((v, i) => {
			const x = padding + i * xStep;
			const y = height - padding - ((v - minV) / range) * (height - padding * 2);
			return `${x.toFixed(1)},${y.toFixed(1)}`;
		}).reverse();

		return `M${topPoints.join(' L')} L${bottomPoints.join(' L')} Z`;
	}

	let chartW = 720;
	let chartH = 280;
	let chartPad = 24;

	let historicalPath = $derived(svgPath(ts?.dates ?? [], ts?.values ?? [], chartW, chartH, chartPad));

	let combinedValues = $derived.by(() => {
		if (!ts || !fc) return [];
		return [...ts.values, ...fc.predictions];
	});
	let combinedMin = $derived(Math.min(...(combinedValues?.filter((v: number) => v != null && isFinite(v)) ?? [0])));
	let combinedMax = $derived(Math.max(...(combinedValues?.filter((v: number) => v != null && isFinite(v)) ?? [1])));
	let combinedRange = $derived(combinedMax - combinedMin || 1);

	function combinedPath(values: number[]): string {
		if (!values?.length) return '';
		const xStep = (chartW - chartPad * 2) / (values.length - 1);
		return values
			.map((v, i) => {
				const x = chartPad + i * xStep;
				const y = chartH - chartPad - ((v - combinedMin) / combinedRange) * (chartH - chartPad * 2);
				return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join(' ');
	}

	function combinedAreaPath(lowers: number[], uppers: number[]): string {
		if (!lowers?.length) return '';
		const totalLen = (ts?.values?.length ?? 0) + lowers.length;
		const xStep = (chartW - chartPad * 2) / (totalLen - 1);
		const startIdx = ts?.values?.length ?? 0;

		const topPoints = uppers.map((v, i) => {
			const x = chartPad + (startIdx + i) * xStep;
			const y = chartH - chartPad - ((v - combinedMin) / combinedRange) * (chartH - chartPad * 2);
			return `${x.toFixed(1)},${y.toFixed(1)}`;
		});
		const bottomPoints = lowers.map((v, i) => {
			const x = chartPad + (startIdx + i) * xStep;
			const y = chartH - chartPad - ((v - combinedMin) / combinedRange) * (chartH - chartPad * 2);
			return `${x.toFixed(1)},${y.toFixed(1)}`;
		}).reverse();

		return `M${topPoints.join(' L')} L${bottomPoints.join(' L')} Z`;
	}

	let histPath = $derived(combinedPath(ts?.values ?? []));
	let fcPath = $derived.by(() => {
		if (!ts || !fc) return '';
		const bridgeValues = [ts.values[ts.values.length - 1], ...fc.predictions];
		const totalLen = ts.values.length + fc.predictions.length;
		const xStep = (chartW - chartPad * 2) / (totalLen - 1);
		const startIdx = ts.values.length - 1;
		return bridgeValues
			.map((v: number, i: number) => {
				const x = chartPad + (startIdx + i) * xStep;
				const y = chartH - chartPad - ((v - combinedMin) / combinedRange) * (chartH - chartPad * 2);
				return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join(' ');
	});
	let ciArea = $derived(combinedAreaPath(fc?.lower ?? [], fc?.upper ?? []));

	let radarFeatures = $derived.by(() => {
		if (!dna?.features) return [];
		const keys = ['trendStrength', 'seasonalStrength', 'forecastability', 'volatility', 'hurstExponent', 'acf1'];
		const labels = ['Trend', 'Seasonal', 'Predictability', 'Volatility', 'Memory', 'Autocorrelation'];
		return keys.map((k, i) => ({
			key: k,
			label: labels[i],
			value: Math.min(1, Math.max(0, dna.features[k] ?? 0))
		}));
	});

	function radarPoints(features: { value: number }[], radius: number, cx: number, cy: number): string {
		return features
			.map((f, i) => {
				const angle = (Math.PI * 2 * i) / features.length - Math.PI / 2;
				const r = f.value * radius;
				return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
			})
			.join(' ');
	}

	function radarAxisEnd(index: number, total: number, radius: number, cx: number, cy: number): { x: number; y: number } {
		const angle = (Math.PI * 2 * index) / total - Math.PI / 2;
		return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
	}

	let radarCx = 140;
	let radarCy = 140;
	let radarR = 110;

	let sortedModels = $derived.by(() => {
		if (!fc?.modelComparison) return [];
		return [...fc.modelComparison].sort((a: any, b: any) => a.mape - b.mape);
	});
	let bestMape = $derived(sortedModels.length > 0 ? sortedModels[0].mape : 1);
</script>

<svelte:head>
	<title>Playground — Vectrix</title>
	<meta name="description" content="Try Vectrix forecasting in your browser. Explore 6 real-world datasets with DNA profiling, automatic model selection, and interactive forecast visualization." />
	<link rel="canonical" href="https://eddmpython.github.io/vectrix/playground" />
	<meta property="og:title" content="Playground — Vectrix" />
	<meta property="og:description" content="Interactive forecasting playground. Explore DNA profiling, model comparison, and predictions on real-world data." />
	<meta property="og:url" content="https://eddmpython.github.io/vectrix/playground" />
</svelte:head>

<Header />

<main class="min-h-screen bg-vx-bg-dark pt-24 pb-16 px-4">
	<div class="max-w-6xl mx-auto">

		<div class="text-center mb-10">
			<Badge variant="accent">
				<Play class="w-3 h-3" />
				Interactive Demo
			</Badge>
			<h1 class="text-3xl md:text-4xl font-extrabold text-vx-text mt-4 mb-3">
				Forecasting <span class="bg-gradient-to-r from-vx-primary to-vx-accent bg-clip-text text-transparent">Playground</span>
			</h1>
			<p class="text-vx-text-muted max-w-xl mx-auto">
				Explore real forecasting results on 6 datasets. No installation needed.
				<br />See how Vectrix analyzes data, profiles patterns, and selects models automatically.
			</p>
		</div>

		<div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-8">
			{#each datasets as ds (ds.id)}
				<button
					class="rounded-xl border p-3 text-left transition-all duration-200 cursor-pointer {selectedId === ds.id
						? 'bg-vx-primary/10 border-vx-primary/50 shadow-lg shadow-vx-primary/10'
						: 'bg-vx-bg-card border-vx-border hover:border-vx-primary/30 hover:bg-vx-bg-card-hover'}"
					onclick={() => (selectedId = ds.id)}
				>
					<div class="text-lg mb-1">{domainIcons[ds.domain] ?? '📊'}</div>
					<div class="text-sm font-semibold text-vx-text truncate">{ds.label}</div>
					<div class="text-xs text-vx-text-dim mt-0.5">{ds.domain}</div>
					<div class="text-xs text-vx-text-dim">{ds.frequency} &middot; {ds.timeSeries.length}pts</div>
				</button>
			{/each}
		</div>

		{#if selected}
			<Card hover={false} class="mb-6 !p-4">
				<div class="flex flex-wrap items-center gap-3 text-sm">
					<span class="text-lg">{domainIcons[selected.domain] ?? '📊'}</span>
					<span class="font-bold text-vx-text text-base">{selected.label}</span>
					<Badge>{selected.frequency}</Badge>
					<Badge variant="accent">{selected.domain}</Badge>
					<span class="text-vx-text-dim">{selected.description}</span>
				</div>
			</Card>

			<div class="flex gap-1 mb-6 bg-vx-bg-card rounded-lg p-1 border border-vx-border w-fit">
				<button
					class="px-4 py-2 rounded-md text-sm font-medium transition-all {activeTab === 'forecast'
						? 'bg-vx-primary/20 text-vx-primary-light'
						: 'text-vx-text-muted hover:text-vx-text'}"
					onclick={() => (activeTab = 'forecast')}
				>
					<TrendingUp class="w-3.5 h-3.5 inline mr-1" />
					Forecast
				</button>
				<button
					class="px-4 py-2 rounded-md text-sm font-medium transition-all {activeTab === 'dna'
						? 'bg-vx-accent/20 text-vx-accent-light'
						: 'text-vx-text-muted hover:text-vx-text'}"
					onclick={() => (activeTab = 'dna')}
				>
					<Dna class="w-3.5 h-3.5 inline mr-1" />
					DNA Profile
				</button>
				<button
					class="px-4 py-2 rounded-md text-sm font-medium transition-all {activeTab === 'models'
						? 'bg-vx-success/20 text-vx-success'
						: 'text-vx-text-muted hover:text-vx-text'}"
					onclick={() => (activeTab = 'models')}
				>
					<Layers class="w-3.5 h-3.5 inline mr-1" />
					Models
				</button>
			</div>

			{#if activeTab === 'forecast'}
				<div class="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">
					<Card hover={false} class="!p-4">
						<div class="text-xs text-vx-text-dim mb-1">Best Model</div>
						<div class="text-sm font-bold text-vx-primary-light truncate">{fc.model}</div>
					</Card>
					<Card hover={false} class="!p-4">
						<div class="text-xs text-vx-text-dim mb-1">MAPE</div>
						<div class="text-xl font-bold text-vx-text">{formatNum(fc.metrics.mape)}%</div>
					</Card>
					<Card hover={false} class="!p-4">
						<div class="text-xs text-vx-text-dim mb-1">Forecast Steps</div>
						<div class="text-xl font-bold text-vx-text">{fc.steps}</div>
					</Card>
					<Card hover={false} class="!p-4">
						<div class="text-xs text-vx-text-dim mb-1">Data Points</div>
						<div class="text-xl font-bold text-vx-text">{ts.length}</div>
					</Card>
				</div>

				<Card hover={false} class="!p-5 mb-6">
					<div class="flex items-center justify-between mb-4">
						<h3 class="text-sm font-semibold text-vx-text">Time Series + Forecast</h3>
						<div class="flex items-center gap-4 text-xs">
							<span class="flex items-center gap-1.5">
								<span class="w-3 h-0.5 bg-vx-primary rounded"></span>
								<span class="text-vx-text-dim">Historical</span>
							</span>
							<span class="flex items-center gap-1.5">
								<span class="w-3 h-0.5 bg-vx-accent rounded"></span>
								<span class="text-vx-text-dim">Forecast</span>
							</span>
							<span class="flex items-center gap-1.5">
								<span class="w-3 h-1.5 bg-vx-accent/20 rounded"></span>
								<span class="text-vx-text-dim">95% CI</span>
							</span>
						</div>
					</div>
					<div class="w-full overflow-x-auto">
						<svg viewBox="0 0 {chartW} {chartH}" class="w-full h-auto min-w-[500px]" preserveAspectRatio="xMidYMid meet">
							<rect x="0" y="0" width={chartW} height={chartH} fill="transparent" />

							{#each [0.25, 0.5, 0.75] as pct}
								<line
									x1={chartPad} x2={chartW - chartPad}
									y1={chartH - chartPad - pct * (chartH - chartPad * 2)}
									y2={chartH - chartPad - pct * (chartH - chartPad * 2)}
									stroke="#334155" stroke-width="0.5" stroke-dasharray="4,4"
								/>
							{/each}

							{#if fc?.lower && fc?.upper}
								<path d={ciArea} fill="#8b5cf6" opacity="0.12" />
							{/if}

							<path d={histPath} fill="none" stroke="#06b6d4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
							<path d={fcPath} fill="none" stroke="#8b5cf6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="6,3" />

							{#if ts?.values?.length}
								{@const totalLen = ts.values.length + (fc?.predictions?.length ?? 0)}
								{@const xStep = (chartW - chartPad * 2) / (totalLen - 1)}
								<line
									x1={chartPad + (ts.values.length - 1) * xStep}
									x2={chartPad + (ts.values.length - 1) * xStep}
									y1={chartPad} y2={chartH - chartPad}
									stroke="#64748b" stroke-width="0.5" stroke-dasharray="3,3"
								/>
							{/if}

							<text x={chartPad + 4} y={chartH - 6} fill="#64748b" font-size="9" font-family="Inter, sans-serif">
								{ts?.dates?.[0]?.slice(0, 7) ?? ''}
							</text>
							<text x={chartW - chartPad - 4} y={chartH - 6} fill="#64748b" font-size="9" font-family="Inter, sans-serif" text-anchor="end">
								{fc?.dates?.[fc.dates.length - 1]?.slice(0, 7) ?? ''}
							</text>

							<text x={chartPad - 2} y={chartPad + 4} fill="#64748b" font-size="9" font-family="Inter, sans-serif" text-anchor="end">
								{formatNum(combinedMax, 0)}
							</text>
							<text x={chartPad - 2} y={chartH - chartPad} fill="#64748b" font-size="9" font-family="Inter, sans-serif" text-anchor="end">
								{formatNum(combinedMin, 0)}
							</text>
						</svg>
					</div>
				</Card>

				<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
					<Card hover={false} class="!p-3 text-center">
						<div class="text-xs text-vx-text-dim">RMSE</div>
						<div class="text-base font-bold text-vx-text mt-1">{formatNum(fc.metrics.rmse)}</div>
					</Card>
					<Card hover={false} class="!p-3 text-center">
						<div class="text-xs text-vx-text-dim">MAE</div>
						<div class="text-base font-bold text-vx-text mt-1">{formatNum(fc.metrics.mae)}</div>
					</Card>
					<Card hover={false} class="!p-3 text-center">
						<div class="text-xs text-vx-text-dim">sMAPE</div>
						<div class="text-base font-bold text-vx-text mt-1">{formatNum(fc.metrics.smape)}%</div>
					</Card>
					<Card hover={false} class="!p-3 text-center">
						<div class="text-xs text-vx-text-dim">Models Compared</div>
						<div class="text-base font-bold text-vx-text mt-1">{fc.modelComparison?.length ?? 0}</div>
					</Card>
				</div>
			{/if}

			{#if activeTab === 'dna'}
				<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
					<Card hover={false} class="!p-5">
						<h3 class="text-sm font-semibold text-vx-text mb-4">DNA Radar</h3>
						<div class="flex justify-center">
							<svg viewBox="0 0 280 280" class="w-full max-w-[280px]">
								{#each [0.25, 0.5, 0.75, 1.0] as ring}
									<polygon
										points={radarFeatures.map((_: any, i: number) => {
											const angle = (Math.PI * 2 * i) / radarFeatures.length - Math.PI / 2;
											const r = ring * radarR;
											return `${radarCx + r * Math.cos(angle)},${radarCy + r * Math.sin(angle)}`;
										}).join(' ')}
										fill="none" stroke="#334155" stroke-width="0.5"
									/>
								{/each}

								{#each radarFeatures as _, i}
									{@const end = radarAxisEnd(i, radarFeatures.length, radarR, radarCx, radarCy)}
									<line x1={radarCx} y1={radarCy} x2={end.x} y2={end.y} stroke="#334155" stroke-width="0.5" />
								{/each}

								<polygon
									points={radarPoints(radarFeatures, radarR, radarCx, radarCy)}
									fill="#8b5cf6" fill-opacity="0.2" stroke="#8b5cf6" stroke-width="1.5"
								/>

								{#each radarFeatures as f, i}
									{@const end = radarAxisEnd(i, radarFeatures.length, radarR + 18, radarCx, radarCy)}
									<text x={end.x} y={end.y} fill="#94a3b8" font-size="9" font-family="Inter, sans-serif" text-anchor="middle" dominant-baseline="middle">
										{f.label}
									</text>
								{/each}
							</svg>
						</div>
					</Card>

					<Card hover={false} class="!p-5">
						<h3 class="text-sm font-semibold text-vx-text mb-4">DNA Summary</h3>
						<div class="space-y-3">
							<div class="flex justify-between items-center">
								<span class="text-xs text-vx-text-dim">Fingerprint</span>
								<code class="text-xs font-mono text-vx-primary-light bg-vx-primary/10 px-2 py-0.5 rounded">{dna.fingerprint}</code>
							</div>
							<div class="flex justify-between items-center">
								<span class="text-xs text-vx-text-dim">Category</span>
								<Badge variant="accent">{dna.category}</Badge>
							</div>
							<div class="flex justify-between items-center">
								<span class="text-xs text-vx-text-dim">Difficulty</span>
								<span class="text-sm font-semibold {difficultyColors[dna.difficulty] ?? 'text-vx-text'}">
									{dna.difficulty.replace('_', ' ')} ({formatNum(dna.difficultyScore, 1)})
								</span>
							</div>
							<div class="flex justify-between items-center">
								<span class="text-xs text-vx-text-dim">Frequency</span>
								<span class="text-sm text-vx-text">{chars?.frequency ?? selected.frequency}</span>
							</div>
							<div class="flex justify-between items-center">
								<span class="text-xs text-vx-text-dim">Period</span>
								<span class="text-sm text-vx-text">{chars?.period ?? '—'}</span>
							</div>
							<div class="flex justify-between items-center">
								<span class="text-xs text-vx-text-dim">Has Trend</span>
								<span class="text-sm {chars?.hasTrend ? 'text-vx-success' : 'text-vx-text-dim'}">{chars?.hasTrend ? 'Yes' : 'No'}</span>
							</div>
							<div class="flex justify-between items-center">
								<span class="text-xs text-vx-text-dim">Has Seasonality</span>
								<span class="text-sm {chars?.hasSeasonality ? 'text-vx-success' : 'text-vx-text-dim'}">{chars?.hasSeasonality ? 'Yes' : 'No'}</span>
							</div>

							<div class="pt-2 border-t border-vx-border">
								<div class="text-xs text-vx-text-dim mb-2">Recommended Models</div>
								<div class="flex flex-wrap gap-1">
									{#each (dna.recommendedModels ?? []).slice(0, 5) as model}
										<span class="text-xs bg-vx-bg-card-hover px-2 py-0.5 rounded text-vx-text-muted">{model}</span>
									{/each}
								</div>
							</div>
						</div>
					</Card>
				</div>

				<Card hover={false} class="!p-5">
					<h3 class="text-sm font-semibold text-vx-text mb-4">Feature Details</h3>
					<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
						{#each Object.entries(dna?.features ?? {}) as [key, value]}
							<div class="bg-vx-bg-darker rounded-lg p-3 border border-vx-border/50">
								<div class="text-xs text-vx-text-dim truncate" title={key}>{key}</div>
								<div class="text-sm font-bold text-vx-text mt-1">{formatNum(value as number, 3)}</div>
								<div class="mt-1.5 h-1 bg-vx-bg-card rounded-full overflow-hidden">
									<div
										class="h-full rounded-full bg-gradient-to-r from-vx-primary to-vx-accent"
										style="width: {Math.min(100, Math.abs((value as number) ?? 0) * 100)}%"
									></div>
								</div>
							</div>
						{/each}
					</div>
				</Card>
			{/if}

			{#if activeTab === 'models'}
				<Card hover={false} class="!p-5 mb-6">
					<h3 class="text-sm font-semibold text-vx-text mb-4">Model Comparison — MAPE (%)</h3>
					<div class="space-y-2">
						{#each sortedModels as model, i (model.model)}
							{@const pct = bestMape > 0 ? Math.min(100, (bestMape / model.mape) * 100) : 100}
							<div class="flex items-center gap-3 group">
								<span class="w-5 text-xs text-vx-text-dim text-right font-mono">{i + 1}</span>
								<div class="flex-1 min-w-0">
									<div class="flex items-center justify-between mb-1">
										<span class="text-sm text-vx-text truncate {i === 0 ? 'font-bold' : ''}">{model.model}</span>
										<span class="text-xs font-mono {i === 0 ? 'text-vx-success font-bold' : 'text-vx-text-muted'}">{formatNum(model.mape)}%</span>
									</div>
									<div class="h-2 bg-vx-bg-darker rounded-full overflow-hidden">
										<div
											class="h-full rounded-full transition-all duration-500 {i === 0 ? 'bg-gradient-to-r from-vx-primary to-vx-accent' : 'bg-vx-bg-card-hover'}"
											style="width: {pct}%"
										></div>
									</div>
								</div>
							</div>
						{/each}
					</div>
				</Card>

				<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
					<Card hover={false} class="!p-5">
						<h3 class="text-sm font-semibold text-vx-text mb-3">Detailed Metrics</h3>
						<div class="overflow-x-auto">
							<table class="w-full text-xs">
								<thead>
									<tr class="border-b border-vx-border">
										<th class="text-left py-2 text-vx-text-dim font-medium">Model</th>
										<th class="text-right py-2 text-vx-text-dim font-medium">MAPE</th>
										<th class="text-right py-2 text-vx-text-dim font-medium">RMSE</th>
										<th class="text-right py-2 text-vx-text-dim font-medium">MAE</th>
									</tr>
								</thead>
								<tbody>
									{#each sortedModels as model, i}
										<tr class="border-b border-vx-border/30 {i === 0 ? 'text-vx-primary-light font-semibold' : 'text-vx-text-muted'}">
											<td class="py-1.5 truncate max-w-[140px]">{model.model}</td>
											<td class="py-1.5 text-right font-mono">{formatNum(model.mape)}</td>
											<td class="py-1.5 text-right font-mono">{formatNum(model.rmse)}</td>
											<td class="py-1.5 text-right font-mono">{formatNum(model.mae)}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					</Card>

					<Card hover={false} class="!p-5">
						<h3 class="text-sm font-semibold text-vx-text mb-3">Execution Time</h3>
						<div class="space-y-2">
							{#each sortedModels as model}
								{@const maxTime = Math.max(...sortedModels.map((m: any) => m.timeMs ?? 0))}
								{@const timePct = maxTime > 0 ? ((model.timeMs ?? 0) / maxTime) * 100 : 0}
								<div class="flex items-center gap-2">
									<span class="text-xs text-vx-text-muted truncate w-28 shrink-0">{model.model}</span>
									<div class="flex-1 h-1.5 bg-vx-bg-darker rounded-full overflow-hidden">
										<div class="h-full rounded-full bg-vx-warning/60" style="width: {timePct}%"></div>
									</div>
									<span class="text-xs text-vx-text-dim font-mono w-14 text-right">{formatNum(model.timeMs ?? 0, 0)}ms</span>
								</div>
							{/each}
						</div>
					</Card>
				</div>
			{/if}

			<div class="mt-10 text-center">
				<p class="text-sm text-vx-text-dim mb-4">Ready to forecast your own data?</p>
				<div class="flex justify-center gap-3">
					<Button href="{base}/docs/getting-started/installation">
						<Download class="w-4 h-4" />
						Install Vectrix
					</Button>
					<Button variant="secondary" href="{base}/docs/getting-started/quickstart">
						<Play class="w-4 h-4" />
						Quickstart
					</Button>
				</div>
			</div>
		{/if}
	</div>
</main>

<Footer />
