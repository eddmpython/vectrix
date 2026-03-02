<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Play, Github } from 'lucide-svelte';
</script>

<section class="relative py-20 px-6 text-center bg-gradient-to-b from-vx-bg-dark to-vx-bg-darker">
	<div class="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-vx-primary to-transparent"></div>

	<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-4">Ready to forecast?</h2>
	<p class="text-vx-text-muted text-lg mb-8">Get started with Vectrix in under a minute.</p>

	<div class="flex justify-center items-center gap-3 flex-wrap">
		<Button href="https://eddmpython.github.io/vectrix/docs/getting-started/quickstart/">
			<Play class="w-4 h-4" />
			Get Started
		</Button>
		<Button variant="secondary" href="https://github.com/eddmpython/vectrix" target="_blank" rel="noopener">
			<Github class="w-4 h-4" />
			Star on GitHub
		</Button>
		<a href="https://buymeacoffee.com/eddmpython" target="_blank" rel="noopener" class="hover:opacity-90 transition-opacity">
			<img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="180" height="45" />
		</a>
	</div>
</section>
