<script lang="ts">
	import { Copy, Check } from 'lucide-svelte';

	const tabs = [
		{ label: 'Default', cmd: 'pip install vectrix' },
		{ label: 'With ML', cmd: 'pip install "vectrix[ml]"' },
		{ label: 'Foundation', cmd: 'pip install "vectrix[foundation]"' },
		{ label: 'Everything', cmd: 'pip install "vectrix[all]"' }
	];

	let activeIdx = $state(0);
	let copied = $state(false);

	function copy() {
		navigator.clipboard.writeText(tabs[activeIdx].cmd);
		copied = true;
		setTimeout(() => copied = false, 2000);
	}
</script>

<section class="py-20 px-6 max-w-3xl mx-auto text-center">
	<span class="text-xs font-semibold uppercase tracking-widest text-vx-primary mb-3 block">Get Started</span>
	<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-3">Install in seconds</h2>
	<p class="text-vx-text-muted max-w-xl mx-auto mb-8">Rust engine included in every wheel — no extras needed.</p>

	<div class="flex justify-center gap-2 mb-6 flex-wrap">
		{#each tabs as tab, i}
			<button
				class="px-4 py-2 rounded-lg text-sm font-medium transition-all cursor-pointer
					{i === activeIdx
						? 'bg-vx-primary text-white'
						: 'bg-transparent border border-vx-border text-vx-text-muted hover:text-vx-text hover:border-vx-primary'}"
				onclick={() => activeIdx = i}
			>
				{tab.label}
			</button>
		{/each}
	</div>

	<div class="flex items-center bg-vx-bg-darker border border-vx-border rounded-xl px-5 py-4 font-mono text-sm">
		<span class="text-vx-primary mr-3 select-none">$</span>
		<span class="text-slate-200 flex-1 text-left">{tabs[activeIdx].cmd}</span>
		<button onclick={copy} class="text-vx-text-dim hover:text-vx-primary transition-colors cursor-pointer p-1">
			{#if copied}
				<Check class="w-4 h-4 text-vx-success" />
			{:else}
				<Copy class="w-4 h-4" />
			{/if}
		</button>
	</div>

	<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8 text-left text-xs text-vx-text-dim">
		<div><strong class="text-vx-text-muted">Default</strong> — 30+ models + Rust engine. All you need.</div>
		<div><strong class="text-vx-text-muted">ML</strong> — LightGBM + XGBoost + scikit-learn.</div>
		<div><strong class="text-vx-text-muted">Foundation</strong> — Chronos-2 + TimesFM zero-shot.</div>
		<div><strong class="text-vx-text-muted">Everything</strong> — All optional extras combined.</div>
	</div>
</section>
