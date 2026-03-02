<section class="bg-vx-bg-darker py-16 px-6">
	<div class="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
		{#each [
			{ value: '200K+', label: 'Time series benchmarked on M3 & M4 competitions' },
			{ value: '65+', label: 'Features extracted by Forecast DNA fingerprinting' },
			{ value: '0.696', label: 'M4 Hourly OWA — competition winner level accuracy' },
			{ value: '<5ms', label: 'ETS model fit with Rust turbo acceleration' }
		] as item}
			<div>
				<span class="block text-4xl font-extrabold tracking-tight bg-gradient-to-r from-vx-primary to-vx-accent bg-clip-text text-transparent mb-1">
					{item.value}
				</span>
				<span class="text-xs text-vx-text-dim leading-snug">{item.label}</span>
			</div>
		{/each}
	</div>
</section>
