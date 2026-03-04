<script lang="ts">
	import { base } from '$app/paths';
	import { Button } from '$lib/components/ui/button';
	import { Play, Github, Coffee } from 'lucide-svelte';
</script>

<section class="relative py-20 px-6 text-center bg-gradient-to-b from-vx-bg-dark to-vx-bg-darker">
	<div class="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-vx-primary to-transparent"></div>

	<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-4">Ready to forecast?</h2>
	<p class="text-vx-text-muted text-lg mb-8">Get started with Vectrix in under a minute.</p>

	<div class="flex justify-center items-center gap-3 flex-wrap">
		<Button href="{base}/docs/getting-started/quickstart">
			<Play class="w-4 h-4" />
			Get Started
		</Button>
		<Button variant="secondary" href="https://github.com/eddmpython/vectrix" target="_blank" rel="noopener">
			<Github class="w-4 h-4" />
			Star on GitHub
		</Button>
		<a href="https://buymeacoffee.com/eddmpython" target="_blank" rel="noopener"
			class="inline-flex items-center gap-2 h-9 px-4 rounded-lg text-sm font-medium bg-[#FFDD00] text-[#0D0C22] hover:bg-[#FFDD00]/90 transition-colors no-underline">
			<Coffee class="w-4 h-4" />
			Buy me a coffee
		</a>
	</div>
</section>
