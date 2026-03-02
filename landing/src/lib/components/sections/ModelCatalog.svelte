<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';

	const categories = [
		{ title: 'Exponential Smoothing', models: ['AutoETS', 'Holt-Winters', 'SES', 'DES', 'AutoCES', '4Theta'], newModels: ['4Theta'] },
		{ title: 'ARIMA Family', models: ['AutoARIMA', 'ARIMAX', 'SARIMA'], newModels: [] },
		{ title: 'Decomposition', models: ['AutoMSTL', 'AutoTBATS', 'Theta', 'DOT'], newModels: [] },
		{ title: 'Non-Parametric', models: ['DTSF', 'ESN', 'Fourier'], newModels: ['DTSF', 'ESN'] },
		{ title: 'Volatility & Special', models: ['GARCH', 'AutoCroston', 'ADIDA', 'IMAPA'], newModels: [] },
		{ title: 'ML & Foundation', models: ['LightGBM', 'XGBoost', 'Chronos-2', 'TimesFM'], newModels: [] }
	];
</script>

<section class="py-20 px-6 max-w-6xl mx-auto text-center">
	<span class="text-xs font-semibold uppercase tracking-widest text-vx-primary mb-3 block">Model Catalog</span>
	<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-3">30+ models, one unified API</h2>
	<p class="text-vx-text-muted max-w-xl mx-auto mb-12">
		Every model shares the same fit → predict interface. Vectrix picks the best one for your data automatically.
	</p>

	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
		{#each categories as cat}
			<Card class="text-left">
				<h3 class="text-sm font-semibold text-vx-text mb-3">{cat.title}</h3>
				<div class="flex flex-wrap gap-1.5">
					{#each cat.models as model}
						{#if cat.newModels.includes(model)}
							<Badge variant="accent">{model}</Badge>
						{:else}
							<Badge>{model}</Badge>
						{/if}
					{/each}
				</div>
			</Card>
		{/each}
	</div>
</section>
