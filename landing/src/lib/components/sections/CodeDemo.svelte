<script lang="ts">
	import CodeWindow from './CodeWindow.svelte';
</script>

<section class="bg-vx-bg-darker py-20 px-6">
	<div class="max-w-6xl mx-auto text-center">
		<span class="text-xs font-semibold uppercase tracking-widest text-vx-primary mb-3 block">How it works</span>
		<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-3">Three lines. Full pipeline.</h2>
		<p class="text-vx-text-muted max-w-xl mx-auto mb-12">
			Forecasting, analysis, and regression — each designed to work with one function call.
		</p>

		<div class="grid grid-cols-1 md:grid-cols-3 gap-5 text-left">
			<CodeWindow title="Forecasting">
				<span class="text-purple-400">from</span> <span class="text-vx-text">vectrix</span> <span class="text-purple-400">import</span> <span class="text-cyan-400">forecast</span><br /><br />
				<span class="text-vx-text">result</span> <span class="text-vx-text-muted">=</span> <span class="text-cyan-400">forecast</span><span class="text-vx-text-muted">(</span><span class="text-emerald-400">"sales.csv"</span><span class="text-vx-text-muted">,</span> <span class="text-vx-text">steps</span><span class="text-vx-text-muted">=</span><span class="text-orange-400">12</span><span class="text-vx-text-muted">)</span><br />
				<span class="text-vx-text">result</span><span class="text-vx-text-muted">.</span><span class="text-cyan-400">plot</span><span class="text-vx-text-muted">()</span><br />
				<span class="text-vx-text">result</span><span class="text-vx-text-muted">.</span><span class="text-cyan-400">to_csv</span><span class="text-vx-text-muted">(</span><span class="text-emerald-400">"forecast.csv"</span><span class="text-vx-text-muted">)</span>
			</CodeWindow>

			<CodeWindow title="Analysis">
				<span class="text-purple-400">from</span> <span class="text-vx-text">vectrix</span> <span class="text-purple-400">import</span> <span class="text-cyan-400">analyze</span><br /><br />
				<span class="text-vx-text">report</span> <span class="text-vx-text-muted">=</span> <span class="text-cyan-400">analyze</span><span class="text-vx-text-muted">(</span><span class="text-emerald-400">"data.csv"</span><span class="text-vx-text-muted">)</span><br />
				<span class="text-cyan-400">print</span><span class="text-vx-text-muted">(</span><span class="text-vx-text">report</span><span class="text-vx-text-muted">.</span><span class="text-vx-text">dna</span><span class="text-vx-text-muted">.</span><span class="text-vx-text">difficulty</span><span class="text-vx-text-muted">)</span><br />
				<span class="text-cyan-400">print</span><span class="text-vx-text-muted">(</span><span class="text-vx-text">report</span><span class="text-vx-text-muted">.</span><span class="text-vx-text">dna</span><span class="text-vx-text-muted">.</span><span class="text-vx-text">recommended_model</span><span class="text-vx-text-muted">)</span>
			</CodeWindow>

			<CodeWindow title="Regression">
				<span class="text-purple-400">from</span> <span class="text-vx-text">vectrix</span> <span class="text-purple-400">import</span> <span class="text-cyan-400">regress</span><br /><br />
				<span class="text-vx-text">r</span> <span class="text-vx-text-muted">=</span> <span class="text-cyan-400">regress</span><span class="text-vx-text-muted">(</span><span class="text-vx-text">df</span><span class="text-vx-text-muted">,</span> <span class="text-emerald-400">"sales ~ ads + price"</span><span class="text-vx-text-muted">)</span><br />
				<span class="text-cyan-400">print</span><span class="text-vx-text-muted">(</span><span class="text-vx-text">r</span><span class="text-vx-text-muted">.</span><span class="text-vx-text">r_squared</span><span class="text-vx-text-muted">)</span><br />
				<span class="text-vx-text">r</span><span class="text-vx-text-muted">.</span><span class="text-cyan-400">plot</span><span class="text-vx-text-muted">()</span>
			</CodeWindow>
		</div>
	</div>
</section>
