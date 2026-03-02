<script lang="ts">
	import { Button } from '$lib/components/ui/button';

	const rows = [
		{ comp: 'M3', yearly: '0.848', quarterly: '0.825', monthly: '0.758', weekly: '—', daily: '—', hourly: '0.819' },
		{ comp: 'M4', yearly: '0.974', quarterly: '0.797', monthly: '0.987', weekly: '0.737', daily: '1.207', hourly: '1.006' },
		{ comp: 'M4 Ensemble', yearly: '0.879', quarterly: '0.797', monthly: '0.927', weekly: '0.737', daily: '1.105', hourly: '0.696' }
	];

	function cellClass(val: string): string {
		if (val === '—') return 'text-vx-text-dim';
		const num = parseFloat(val);
		if (isNaN(num)) return '';
		return num < 1.0 ? 'text-vx-success font-semibold' : 'text-vx-warning font-semibold';
	}
</script>

<section class="bg-vx-bg-darker py-20 px-6">
	<div class="max-w-5xl mx-auto text-center">
		<span class="text-xs font-semibold uppercase tracking-widest text-vx-primary mb-3 block">Benchmarks</span>
		<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-3">M3 & M4 Competition Results</h2>
		<p class="text-vx-text-muted max-w-xl mx-auto mb-12">
			Tested against 200,000+ real-world time series. OWA &lt; 1.0 means beating the Naive2 baseline.
		</p>

		<div class="rounded-xl overflow-hidden bg-vx-bg-card border border-vx-border">
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead>
						<tr class="border-b-2 border-vx-border">
							<th class="px-4 py-3 text-left text-xs uppercase tracking-wider text-vx-text-dim font-semibold">Competition</th>
							{#each ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly'] as h}
								<th class="px-4 py-3 text-center text-xs uppercase tracking-wider text-vx-text-dim font-semibold">{h}</th>
							{/each}
						</tr>
					</thead>
					<tbody>
						{#each rows as row}
							<tr class="border-b border-vx-border/50">
								<td class="px-4 py-3 text-left font-semibold text-vx-text">{row.comp}</td>
								{#each [row.yearly, row.quarterly, row.monthly, row.weekly, row.daily, row.hourly] as val}
									<td class="px-4 py-3 text-center {cellClass(val)}">{val}</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>

		<p class="mt-3 text-xs text-vx-text-dim">
			M4 Ensemble uses VX-Ensemble with DOT + AutoCES + 4Theta + DTSF + ESN. Hourly 0.696 OWA = competition winner level.
		</p>
		<Button variant="secondary" size="sm" href="https://eddmpython.github.io/vectrix/docs/benchmarks/" class="mt-6">
			View full benchmark results →
		</Button>
	</div>
</section>
