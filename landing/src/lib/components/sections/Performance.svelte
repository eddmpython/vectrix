<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';

	const benchmarks = [
		{ name: 'AutoETS', python: 348, rust: 32, speedup: '10.8x' },
		{ name: 'DOT', python: 68, rust: 2.8, speedup: '24x' },
		{ name: 'AutoCES', python: 118, rust: 9.6, speedup: '12x' },
		{ name: 'AutoARIMA', python: 285, rust: 51, speedup: '5.6x' },
		{ name: '4Theta', python: 63, rust: 5.6, speedup: '11x' },
		{ name: 'Theta', python: 130, rust: 16, speedup: '8.1x' }
	];
</script>

<section class="py-20 px-6 max-w-5xl mx-auto text-center">
	<span class="text-xs font-semibold uppercase tracking-widest text-vx-primary mb-3 block">Performance</span>
	<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-3">Built-in Rust Engine</h2>
	<p class="text-vx-text-muted max-w-xl mx-auto mb-12">
		29 Rust-accelerated core loops compiled into every wheel. Like Polars — <code class="text-vx-primary">pip install</code> and it's fast.
	</p>

	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
		{#each benchmarks as b}
			<Card class="text-left" hover={false}>
				<div class="flex items-center justify-between mb-4">
					<span class="font-semibold text-vx-text">{b.name}</span>
					<Badge variant="success">{b.speedup} faster</Badge>
				</div>
				<div class="flex flex-col gap-2">
					<div class="relative h-7 rounded-md bg-vx-text-dim/20 overflow-hidden">
						<div class="absolute inset-y-0 left-0 bg-vx-text-dim/30 rounded-md" style="width: 100%"></div>
						<span class="absolute inset-y-0 left-3 flex items-center text-xs text-vx-text-muted">Python: {b.python}ms</span>
					</div>
					<div class="relative h-7 rounded-md overflow-hidden">
						<div class="absolute inset-y-0 left-0 bg-gradient-to-r from-vx-primary to-vx-accent rounded-md"
							style="width: {Math.max((b.rust / b.python) * 100, 8)}%"></div>
						<span class="absolute inset-y-0 left-3 flex items-center text-xs text-white font-medium">Rust: {b.rust}ms</span>
					</div>
				</div>
			</Card>
		{/each}
	</div>

	<p class="mt-8 text-sm text-vx-text-dim">
		<code class="px-2 py-1 rounded bg-vx-bg-card text-vx-primary text-xs">pip install vectrix</code>
		— Rust engine included in every wheel. No extras, no flags needed.
	</p>
</section>
