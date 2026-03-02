<script lang="ts">
	import { Check, X } from 'lucide-svelte';

	const capabilities = [
		{ name: 'Zero-config forecasting', vectrix: true, statsforecast: true, prophet: false, darts: false },
		{ name: 'Pure Python (no compiled deps)', vectrix: true, statsforecast: false, prophet: false, darts: false },
		{ name: '30+ statistical models', vectrix: true, statsforecast: true, prophet: false, darts: true },
		{ name: 'Flat prediction defense', vectrix: true, statsforecast: false, prophet: false, darts: false },
		{ name: 'Forecast DNA fingerprinting', vectrix: true, statsforecast: false, prophet: false, darts: false },
		{ name: 'Business constraints (8 types)', vectrix: true, statsforecast: false, prophet: false, darts: false },
		{ name: 'R-style regression', vectrix: true, statsforecast: false, prophet: false, darts: false },
		{ name: 'Self-healing forecasts', vectrix: true, statsforecast: false, prophet: false, darts: false },
		{ name: 'Foundation model wrappers', vectrix: true, statsforecast: false, prophet: false, darts: true },
		{ name: 'VAR / VECM multivariate', vectrix: true, statsforecast: false, prophet: false, darts: true }
	];
</script>

<section class="bg-vx-bg-darker py-20 px-6">
	<div class="max-w-5xl mx-auto text-center">
		<span class="text-xs font-semibold uppercase tracking-widest text-vx-primary mb-3 block">Comparison</span>
		<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-3">How Vectrix compares</h2>
		<p class="text-vx-text-muted max-w-xl mx-auto mb-12">
			Purpose-built for production forecasting with features no other library offers.
		</p>

		<div class="rounded-xl overflow-hidden bg-vx-bg-card border border-vx-border">
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead>
						<tr class="border-b-2 border-vx-border">
							<th class="px-4 py-3 text-left text-xs uppercase tracking-wider text-vx-text-dim font-semibold">Capability</th>
							<th class="px-4 py-3 text-center text-xs uppercase tracking-wider text-vx-primary font-bold">Vectrix</th>
							<th class="px-4 py-3 text-center text-xs uppercase tracking-wider text-vx-text-dim font-semibold">statsforecast</th>
							<th class="px-4 py-3 text-center text-xs uppercase tracking-wider text-vx-text-dim font-semibold">Prophet</th>
							<th class="px-4 py-3 text-center text-xs uppercase tracking-wider text-vx-text-dim font-semibold">Darts</th>
						</tr>
					</thead>
					<tbody>
						{#each capabilities as cap}
							<tr class="border-b border-vx-border/50">
								<td class="px-4 py-3 text-left text-vx-text">{cap.name}</td>
								<td class="px-4 py-3 text-center">
									{#if cap.vectrix}<Check class="w-4 h-4 text-vx-success mx-auto" />{:else}<X class="w-4 h-4 text-vx-text-dim mx-auto" />{/if}
								</td>
								<td class="px-4 py-3 text-center">
									{#if cap.statsforecast}<Check class="w-4 h-4 text-vx-success mx-auto" />{:else}<X class="w-4 h-4 text-vx-text-dim mx-auto" />{/if}
								</td>
								<td class="px-4 py-3 text-center">
									{#if cap.prophet}<Check class="w-4 h-4 text-vx-success mx-auto" />{:else}<X class="w-4 h-4 text-vx-text-dim mx-auto" />{/if}
								</td>
								<td class="px-4 py-3 text-center">
									{#if cap.darts}<Check class="w-4 h-4 text-vx-success mx-auto" />{:else}<X class="w-4 h-4 text-vx-text-dim mx-auto" />{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
	</div>
</section>
