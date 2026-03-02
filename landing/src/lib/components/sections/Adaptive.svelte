<script lang="ts">
	import { Eye, CheckCircle, LayoutGrid, Wrench } from 'lucide-svelte';
	import CodeWindow from './CodeWindow.svelte';

	const items = [
		{ icon: Eye, color: 'text-vx-primary bg-vx-primary/10', title: 'Regime Detection', desc: 'HMM-based automatic regime identification. Detect structural breaks and adapt model selection accordingly.' },
		{ icon: CheckCircle, color: 'text-vx-success bg-vx-success/10', title: 'Self-Healing Forecasts', desc: 'CUSUM + EWMA monitoring. When predictions drift, the system automatically corrects before errors accumulate.' },
		{ icon: LayoutGrid, color: 'text-vx-accent bg-vx-accent/10', title: '8 Business Constraints', desc: 'Non-negative, bounded range, monotonic, integer-only, seasonal floor/ceiling, growth rate limits, and more.' },
		{ icon: Wrench, color: 'text-vx-warning bg-vx-warning/10', title: 'Forecast DNA Fingerprinting', desc: '65+ features: trend strength, seasonality type, noise ratio, Hurst exponent, demand density, and more.' }
	];
</script>

<section class="bg-vx-bg-darker py-20 px-6">
	<div class="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
		<div>
			<span class="text-xs font-semibold uppercase tracking-widest text-vx-primary mb-3 block">Adaptive Intelligence</span>
			<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-4">Forecasts that adapt to reality</h2>
			<p class="text-vx-text-muted mb-8">
				Real-world data is messy. Regimes shift, anomalies appear, business rules matter. Vectrix handles it all.
			</p>
			<div class="flex flex-col gap-5">
				{#each items as item}
					<div class="flex gap-4 items-start">
						<div class="w-9 h-9 min-w-9 rounded-lg flex items-center justify-center {item.color}">
							<item.icon class="w-4 h-4" />
						</div>
						<div>
							<strong class="text-sm text-vx-text">{item.title}</strong>
							<p class="text-xs text-vx-text-muted leading-relaxed mt-0.5">{item.desc}</p>
						</div>
					</div>
				{/each}
			</div>
		</div>

		<div class="lg:sticky lg:top-20">
			<CodeWindow title="adaptive.py">
				<span class="text-purple-400">from</span> <span class="text-vx-text">vectrix</span> <span class="text-purple-400">import</span> <span class="text-cyan-400">Vectrix</span><br /><br />
				<span class="text-vx-text">vx</span> <span class="text-vx-text-muted">=</span> <span class="text-cyan-400">Vectrix</span><span class="text-vx-text-muted">(</span><span class="text-vx-text">data</span><span class="text-vx-text-muted">)</span><br /><br />
				<span class="text-vx-text-dim"># regime detection</span><br />
				<span class="text-vx-text">regimes</span> <span class="text-vx-text-muted">=</span> <span class="text-vx-text">vx</span><span class="text-vx-text-muted">.</span><span class="text-cyan-400">detectRegimes</span><span class="text-vx-text-muted">()</span><br /><br />
				<span class="text-vx-text-dim"># self-healing forecast</span><br />
				<span class="text-vx-text">healed</span> <span class="text-vx-text-muted">=</span> <span class="text-vx-text">vx</span><span class="text-vx-text-muted">.</span><span class="text-cyan-400">healForecast</span><span class="text-vx-text-muted">(</span><br />
				&nbsp;&nbsp;<span class="text-vx-text">steps</span><span class="text-vx-text-muted">=</span><span class="text-orange-400">12</span><span class="text-vx-text-muted">,</span><br />
				&nbsp;&nbsp;<span class="text-vx-text">actual</span><span class="text-vx-text-muted">=</span><span class="text-vx-text">newData</span><br />
				<span class="text-vx-text-muted">)</span><br /><br />
				<span class="text-vx-text-dim"># business constraints</span><br />
				<span class="text-vx-text">result</span> <span class="text-vx-text-muted">=</span> <span class="text-vx-text">vx</span><span class="text-vx-text-muted">.</span><span class="text-cyan-400">forecast</span><span class="text-vx-text-muted">(</span><br />
				&nbsp;&nbsp;<span class="text-vx-text">steps</span><span class="text-vx-text-muted">=</span><span class="text-orange-400">12</span><span class="text-vx-text-muted">,</span><br />
				&nbsp;&nbsp;<span class="text-vx-text">constraints</span><span class="text-vx-text-muted">=[</span><br />
				&nbsp;&nbsp;&nbsp;&nbsp;<span class="text-cyan-400">NonNegative</span><span class="text-vx-text-muted">(),</span><br />
				&nbsp;&nbsp;&nbsp;&nbsp;<span class="text-cyan-400">MaxGrowth</span><span class="text-vx-text-muted">(</span><span class="text-orange-400">0.1</span><span class="text-vx-text-muted">)</span><br />
				&nbsp;&nbsp;<span class="text-vx-text-muted">]</span><br />
				<span class="text-vx-text-muted">)</span>
			</CodeWindow>
		</div>
	</div>
</section>
