<script lang="ts">
	import Header from '$lib/components/sections/Header.svelte';
	import Hero from '$lib/components/sections/Hero.svelte';
	import Numbers from '$lib/components/sections/Numbers.svelte';
	import Features from '$lib/components/sections/Features.svelte';
	import CodeDemo from '$lib/components/sections/CodeDemo.svelte';
	import ModelCatalog from '$lib/components/sections/ModelCatalog.svelte';
	import Adaptive from '$lib/components/sections/Adaptive.svelte';
	import Performance from '$lib/components/sections/Performance.svelte';
	import Benchmarks from '$lib/components/sections/Benchmarks.svelte';
	import Comparison from '$lib/components/sections/Comparison.svelte';
	import UseCases from '$lib/components/sections/UseCases.svelte';
	import Install from '$lib/components/sections/Install.svelte';
	import CTA from '$lib/components/sections/CTA.svelte';
	import Footer from '$lib/components/sections/Footer.svelte';
</script>

<svelte:head>
	<title>Vectrix — Zero-Config Time Series Forecasting for Python</title>
</svelte:head>

<Header />
<main class="overflow-x-hidden">
	<Hero />
	<Numbers />
	<Features />
	<CodeDemo />
	<ModelCatalog />
	<Adaptive />
	<Performance />
	<Benchmarks />
	<Comparison />
	<UseCases />
	<Install />
	<CTA />
	<Footer />
</main>
