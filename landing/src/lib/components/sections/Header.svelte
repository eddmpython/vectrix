<script lang="ts">
	import { base } from '$app/paths';
	import { Button } from '$lib/components/ui/button';
	import { Github, BookOpen, Download } from 'lucide-svelte';

	let scrolled = $state(false);

	function handleScroll() {
		scrolled = window.scrollY > 20;
	}
</script>

<svelte:window onscroll={handleScroll} />

<header class="fixed top-0 left-0 right-0 z-50 transition-all duration-300 {scrolled ? 'bg-vx-bg-darker/90 backdrop-blur-xl border-b border-vx-border/50 shadow-lg shadow-black/20' : 'bg-transparent'}">
	<nav class="max-w-6xl mx-auto flex items-center justify-between px-6 h-16">
		<a href="{base}/" class="flex items-center gap-1.5 no-underline group">
			<img src="{base}/icon-final.png" alt="Vectrix" width="28" height="28" class="transition-transform group-hover:scale-110" />
			<span class="text-lg font-bold text-vx-text tracking-tight">Vectrix</span>
		</a>

		<div class="hidden md:flex items-center gap-1">
			<a href="https://eddmpython.github.io/vectrix/docs/getting-started/quickstart/"
				class="px-3 py-2 text-sm text-vx-text-muted hover:text-vx-text transition-colors no-underline rounded-lg hover:bg-white/5">
				Quickstart
			</a>
			<a href="https://eddmpython.github.io/vectrix/docs/api/easy/"
				class="px-3 py-2 text-sm text-vx-text-muted hover:text-vx-text transition-colors no-underline rounded-lg hover:bg-white/5">
				API
			</a>
			<a href="https://eddmpython.github.io/vectrix/docs/guide/forecasting/"
				class="px-3 py-2 text-sm text-vx-text-muted hover:text-vx-text transition-colors no-underline rounded-lg hover:bg-white/5">
				Guide
			</a>
			<a href="https://eddmpython.github.io/vectrix/docs/benchmarks/"
				class="px-3 py-2 text-sm text-vx-text-muted hover:text-vx-text transition-colors no-underline rounded-lg hover:bg-white/5">
				Benchmarks
			</a>
		</div>

		<div class="flex items-center gap-2">
			<a href="https://github.com/eddmpython/vectrix" target="_blank" rel="noopener"
				class="w-9 h-9 rounded-lg flex items-center justify-center text-vx-text-dim hover:text-vx-text hover:bg-white/5 transition-all no-underline"
				title="GitHub">
				<Github class="w-4.5 h-4.5" />
			</a>
			<a href="https://eddmpython.github.io/vectrix/docs/" target="_blank" rel="noopener"
				class="w-9 h-9 rounded-lg flex items-center justify-center text-vx-text-dim hover:text-vx-text hover:bg-white/5 transition-all no-underline"
				title="Documentation">
				<BookOpen class="w-4.5 h-4.5" />
			</a>
			<Button size="sm" href="https://eddmpython.github.io/vectrix/docs/getting-started/installation/">
				<Download class="w-3.5 h-3.5" />
				Install
			</Button>
		</div>
	</nav>
</header>
