<script lang="ts">
	import { base } from '$app/paths';
	import { navigation, type NavItem } from '$lib/docs/navigation';
	import { locale, type Locale } from '$lib/docs/i18n';
	import { BookOpen, Code2, Layers, Sparkles, Briefcase, GitBranch, BarChart3, FileText, Rocket, GraduationCap, FlaskConical, Activity } from 'lucide-svelte';

	let currentLang: Locale = $state('en');
	locale.subscribe(v => currentLang = v);

	function t(item: NavItem) {
		return currentLang === 'ko' && item.titleKo ? item.titleKo : item.title;
	}

	const sectionIcons: Record<string, any> = {
		'Getting Started': Rocket,
		'User Guide': BookOpen,
		'API Reference': Code2,
		'Tutorials': GraduationCap,
		'Showcase': FlaskConical,
		'Benchmarks': BarChart3,
		'Changelog': FileText,
	};

	const sectionDescriptions: Record<string, { en: string; ko: string }> = {
		'Getting Started': { en: 'Install and run your first forecast in 30 seconds', ko: '30\ucd08 \uc548\uc5d0 \uc124\uce58\ud558\uace0 \uccab \uc608\uce21\uc744 \uc2e4\ud589\ud558\uc138\uc694' },
		'User Guide': { en: 'Deep-dive into forecasting, analysis, regression, and more', ko: '\uc608\uce21, \ubd84\uc11d, \ud68c\uadc0\ubd84\uc11d \ub4f1 \uc2ec\uce35 \uac00\uc774\ub4dc' },
		'API Reference': { en: 'Every class, function, and parameter documented', ko: '\ubaa8\ub4e0 \ud074\ub798\uc2a4, \ud568\uc218, \ub9e4\uac1c\ubcc0\uc218 \ubb38\uc11c' },
		'Tutorials': { en: 'Step-by-step walkthroughs from basics to advanced', ko: '\uae30\ucd08\ubd80\ud130 \uace0\uae09\uae4c\uc9c0 \ub2e8\uacc4\ubcc4 \ud29c\ud1a0\ub9ac\uc5bc' },
		'Showcase': { en: 'Real-world examples with Korean economic data', ko: '\ud55c\uad6d \uacbd\uc81c \ub370\uc774\ud130\ub85c \ubcf4\ub294 \uc2e4\uc804 \uc608\uc2dc' },
		'Benchmarks': { en: 'M3/M4 competition results and speed comparisons', ko: 'M3/M4 \ub300\ud68c \uacb0\uacfc \ubc0f \uc18d\ub3c4 \ube44\uad50' },
		'Changelog': { en: 'Version history and release notes', ko: '\ubc84\uc804 \uc774\ub825 \ubc0f \ub9b4\ub9ac\uc2a4 \ub178\ud2b8' },
	};
</script>

<svelte:head>
	<title>Documentation — Vectrix</title>
	<meta name="description" content="Vectrix documentation — guides, API reference, tutorials, and benchmarks for time series forecasting in Python." />
	<link rel="canonical" href="https://eddmpython.github.io/vectrix/docs" />

	<meta property="og:type" content="website" />
	<meta property="og:title" content="Documentation — Vectrix" />
	<meta property="og:description" content="Guides, API reference, tutorials, and benchmarks for time series forecasting in Python." />
	<meta property="og:url" content="https://eddmpython.github.io/vectrix/docs" />
	<meta property="og:image" content="https://eddmpython.github.io/vectrix/icon-512.png" />
	<meta property="og:site_name" content="Vectrix" />
</svelte:head>

<div class="docs-index">
	<div class="docs-hero">
		<h1>{currentLang === 'ko' ? 'Vectrix \ubb38\uc11c' : 'Vectrix Documentation'}</h1>
		<p>{currentLang === 'ko' ? '\ud55c \uc904\uc758 \ucf54\ub4dc\ub85c \uc2dc\uc791\ud558\ub294 \uc2dc\uacc4\uc5f4 \uc608\uce21' : 'Zero-config time series forecasting for Python'}</p>
	</div>

	<div class="docs-grid">
		{#each navigation as section}
			{@const Icon = sectionIcons[section.title] ?? BookOpen}
			{@const desc = sectionDescriptions[section.title]}
			{@const href = section.items?.[0]?.href ?? section.href}
			<a href="{base}{href}" class="docs-card">
				<div class="docs-card-icon">
					<Icon size={22} />
				</div>
				<div class="docs-card-body">
					<h3>{t(section)}</h3>
					<p>{desc ? (currentLang === 'ko' ? desc.ko : desc.en) : ''}</p>
					{#if section.items}
						<span class="docs-card-count">{section.items.length} {currentLang === 'ko' ? '\ud398\uc774\uc9c0' : 'pages'}</span>
					{/if}
				</div>
			</a>
		{/each}
	</div>

	<div class="docs-quickstart">
		<pre><code>pip install vectrix</code></pre>
		<pre><code>from vectrix import forecast{'\n'}result = forecast(data, steps=30){'\n'}print(result.summary())</code></pre>
	</div>
</div>

<style>
	.docs-index {
		max-width: 860px;
		margin: 0 auto;
	}

	.docs-hero {
		text-align: center;
		padding: 2rem 0 2.5rem;
	}

	.docs-hero h1 {
		font-size: 2.2rem;
		font-weight: 800;
		background: linear-gradient(135deg, #f8fafc 30%, #06b6d4);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		margin-bottom: 0.5rem;
	}

	.docs-hero p {
		color: #64748b;
		font-size: 1rem;
	}

	/* Grid */
	.docs-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
		gap: 0.75rem;
	}

	.docs-card {
		display: flex;
		gap: 0.85rem;
		padding: 1rem 1.15rem;
		border-radius: 10px;
		border: 1px solid rgba(148, 163, 184, 0.08);
		background: rgba(15, 23, 42, 0.5);
		text-decoration: none;
		transition: all 0.2s;
	}

	.docs-card:hover {
		border-color: rgba(6, 182, 212, 0.3);
		background: rgba(6, 182, 212, 0.04);
		transform: translateY(-1px);
	}

	.docs-card-icon {
		flex-shrink: 0;
		width: 40px;
		height: 40px;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 8px;
		background: rgba(6, 182, 212, 0.08);
		color: #06b6d4;
	}

	.docs-card-body h3 {
		font-size: 0.92rem;
		font-weight: 600;
		color: #f8fafc;
		margin: 0 0 0.2rem;
	}

	.docs-card-body p {
		font-size: 0.78rem;
		color: #64748b;
		line-height: 1.4;
		margin: 0;
	}

	.docs-card-count {
		font-size: 0.7rem;
		color: #475569;
		margin-top: 0.3rem;
		display: inline-block;
	}

	/* Quick install */
	.docs-quickstart {
		margin-top: 2.5rem;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.docs-quickstart pre {
		background: #0d1117;
		border: 1px solid rgba(148, 163, 184, 0.1);
		border-radius: 8px;
		padding: 0.75rem 1rem;
		overflow-x: auto;
		margin: 0;
	}

	.docs-quickstart code {
		font-family: 'JetBrains Mono', monospace;
		font-size: 0.82rem;
		color: #e2e8f0;
	}

	@media (max-width: 560px) {
		.docs-grid {
			grid-template-columns: 1fr;
		}
		.docs-hero h1 { font-size: 1.6rem; }
	}
</style>
