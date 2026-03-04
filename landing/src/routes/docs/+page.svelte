<script lang="ts">
	import { base } from '$app/paths';
	import { navigation, type NavItem } from '$lib/docs/navigation';
	import { BookOpen, Code2, Layers, Sparkles, Briefcase, GitBranch, BarChart3, FileText, Rocket, GraduationCap, Activity } from 'lucide-svelte';

	const sectionIcons: Record<string, any> = {
		'Getting Started': Rocket,
		'User Guide': BookOpen,
		'API Reference': Code2,
		'Tutorials': GraduationCap,
		'Blog': FileText,
		'Benchmarks': BarChart3,
		'Changelog': FileText,
	};

	const sectionDescriptions: Record<string, string> = {
		'Getting Started': 'Install and run your first forecast in 30 seconds',
		'User Guide': 'Deep-dive into forecasting, analysis, regression, and more',
		'API Reference': 'Every class, function, and parameter documented',
		'Tutorials': 'Step-by-step walkthroughs from basics to advanced',
		'Blog': 'Learn forecasting from fundamentals to advanced techniques',
		'Benchmarks': 'M3/M4 competition results and speed comparisons',
		'Changelog': 'Version history and release notes',
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
		<h1>Vectrix Documentation</h1>
		<p>Zero-config time series forecasting for Python</p>
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
					<h3>{section.title}</h3>
					<p>{desc ?? ''}</p>
					{#if section.items}
						<span class="docs-card-count">{section.items.length} pages</span>
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
