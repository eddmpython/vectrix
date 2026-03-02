<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Zap, Download, Play, Github } from 'lucide-svelte';
	import CodeWindow from './CodeWindow.svelte';
</script>

<section class="relative px-6 pt-24 pb-20 text-center overflow-hidden">
	<div class="absolute inset-0 bg-gradient-to-b from-vx-bg-darker to-vx-bg-dark"></div>
	<div class="absolute -top-1/2 -left-1/2 w-[200%] h-[200%] animate-pulse-slow pointer-events-none"
		style="background: radial-gradient(ellipse at 30% 50%, rgba(6,182,212,0.08) 0%, transparent 50%), radial-gradient(ellipse at 70% 50%, rgba(139,92,246,0.06) 0%, transparent 50%);">
	</div>

	<div class="relative z-10 max-w-4xl mx-auto">
		<Badge class="mb-6">
			<Zap class="w-3.5 h-3.5" />
			v0.0.5 — Rust Turbo Mode Available
		</Badge>

		<h1 class="text-5xl md:text-7xl font-extrabold tracking-tight leading-[1.1] mb-4">
			<span class="bg-gradient-to-br from-vx-text to-vx-text-muted bg-clip-text text-transparent">
				Time Series Forecasting
			</span>
			<br />
			<span class="bg-gradient-to-r from-vx-primary to-vx-accent bg-clip-text text-transparent">
				Made Simple
			</span>
		</h1>

		<p class="text-lg md:text-xl text-vx-text-muted font-light max-w-xl mx-auto mb-8 leading-relaxed">
			One line of code. 30+ models. Zero configuration.<br />
			Pure Python — no compiled dependencies required.
		</p>

		<div class="flex justify-center gap-6 mb-10 flex-wrap">
			{#each [
				{ value: '30+', label: 'Models' },
				{ value: '3', label: 'Dependencies' },
				{ value: '24x', label: 'Rust Turbo' },
				{ value: '573', label: 'Tests' }
			] as stat}
				<div class="text-center">
					<span class="block text-3xl font-bold bg-gradient-to-r from-vx-primary to-vx-accent bg-clip-text text-transparent">
						{stat.value}
					</span>
					<span class="text-xs text-vx-text-dim uppercase tracking-wider">{stat.label}</span>
				</div>
			{/each}
		</div>

		<div class="flex justify-center gap-3 mb-12 flex-wrap">
			<Button href="https://eddmpython.github.io/vectrix/docs/getting-started/installation/">
				<Download class="w-4 h-4" />
				Install
			</Button>
			<Button variant="secondary" href="https://eddmpython.github.io/vectrix/docs/getting-started/quickstart/">
				<Play class="w-4 h-4" />
				Quickstart
			</Button>
			<Button variant="secondary" href="https://github.com/eddmpython/vectrix" target="_blank" rel="noopener">
				<Github class="w-4 h-4" />
				GitHub
			</Button>
		</div>

		<CodeWindow title="forecast.py">
			<span class="text-purple-400">from</span> <span class="text-vx-text">vectrix</span> <span class="text-purple-400">import</span> <span class="text-cyan-400">forecast</span><br /><br />
			<span class="text-vx-text">result</span> <span class="text-vx-text-muted">=</span> <span class="text-cyan-400">forecast</span><span class="text-vx-text-muted">(</span><span class="text-emerald-400">"sales.csv"</span><span class="text-vx-text-muted">,</span> <span class="text-vx-text">steps</span><span class="text-vx-text-muted">=</span><span class="text-orange-400">12</span><span class="text-vx-text-muted">)</span><br />
			<span class="text-vx-text">result</span><span class="text-vx-text-muted">.</span><span class="text-cyan-400">plot</span><span class="text-vx-text-muted">()</span>  <span class="text-vx-text-dim"># auto model + intervals + plot</span>
		</CodeWindow>
	</div>
</section>
