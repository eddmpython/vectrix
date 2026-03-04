<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';

	let container: HTMLDivElement | undefined = $state();

	function loadGiscus() {
		if (!container) return;
		container.innerHTML = '';

		const script = document.createElement('script');
		script.src = 'https://giscus.app/client.js';
		script.setAttribute('data-repo', 'eddmpython/vectrix');
		script.setAttribute('data-repo-id', 'R_kgDORZ7Y3A');
		script.setAttribute('data-category', 'General');
		script.setAttribute('data-category-id', 'DIC_kwDORZ7Y3M4C3pFA');
		script.setAttribute('data-mapping', 'pathname');
		script.setAttribute('data-strict', '0');
		script.setAttribute('data-reactions-enabled', '1');
		script.setAttribute('data-emit-metadata', '0');
		script.setAttribute('data-input-position', 'top');
		script.setAttribute('data-theme', 'noborder_dark');
		script.setAttribute('data-lang', 'en');
		script.setAttribute('data-loading', 'lazy');
		script.crossOrigin = 'anonymous';
		script.async = true;
		container.appendChild(script);
	}

	onMount(() => {
		loadGiscus();
	});

	$effect(() => {
		$page.url.pathname;
		loadGiscus();
	});
</script>

<div class="giscus-wrapper" bind:this={container}></div>

<style>
	.giscus-wrapper {
		margin-top: 3rem;
		padding-top: 2rem;
		border-top: 1px solid rgba(148, 163, 184, 0.1);
	}
</style>
