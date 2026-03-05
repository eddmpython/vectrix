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
	<meta name="description" content="Zero-config time series forecasting for Python. 30+ models, one line of code, built-in Rust engine. Automatic model selection, confidence intervals, and business intelligence." />
	<link rel="canonical" href="https://eddmpython.github.io/vectrix/" />

	<meta property="og:type" content="website" />
	<meta property="og:title" content="Vectrix — Zero-Config Time Series Forecasting for Python" />
	<meta property="og:description" content="Production-ready time series forecasting library. 30+ statistical models, automatic selection, built-in Rust engine." />
	<meta property="og:url" content="https://eddmpython.github.io/vectrix/" />
	<meta property="og:image" content="https://eddmpython.github.io/vectrix/icon-512.png" />
	<meta property="og:site_name" content="Vectrix" />

	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content="Vectrix — Zero-Config Time Series Forecasting for Python" />
	<meta name="twitter:description" content="Production-ready time series forecasting. 30+ models, one line of code." />
	<meta name="twitter:image" content="https://eddmpython.github.io/vectrix/icon-512.png" />

	{@html `<script type="application/ld+json">${JSON.stringify({
		"@context": "https://schema.org",
		"@type": "SoftwareApplication",
		"name": "Vectrix",
		"applicationCategory": "DeveloperApplication",
		"operatingSystem": "Windows, macOS, Linux",
		"description": "Zero-config time series forecasting for Python. 30+ models, one line of code, built-in Rust engine.",
		"url": "https://eddmpython.github.io/vectrix/",
		"downloadUrl": "https://pypi.org/project/vectrix/",
		"softwareVersion": "0.0.16",
		"author": { "@type": "Person", "name": "eddmpython", "url": "https://github.com/eddmpython" },
		"offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" },
		"license": "https://opensource.org/licenses/MIT",
		"programmingLanguage": "Python",
		"codeRepository": "https://github.com/eddmpython/vectrix"
	})}</script>`}

	{@html `<script type="application/ld+json">${JSON.stringify({
		"@context": "https://schema.org",
		"@type": "Organization",
		"name": "Vectrix",
		"url": "https://eddmpython.github.io/vectrix/",
		"logo": "https://eddmpython.github.io/vectrix/icon-512.png",
		"sameAs": [
			"https://github.com/eddmpython/vectrix",
			"https://pypi.org/project/vectrix/"
		]
	})}</script>`}

	{@html `<script type="application/ld+json">${JSON.stringify({
		"@context": "https://schema.org",
		"@type": "WebSite",
		"name": "Vectrix",
		"url": "https://eddmpython.github.io/vectrix/",
		"potentialAction": {
			"@type": "SearchAction",
			"target": "https://eddmpython.github.io/vectrix/docs/api/easy",
			"query-input": "required name=search_term_string"
		}
	})}</script>`}
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
