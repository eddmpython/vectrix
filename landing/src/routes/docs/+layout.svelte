<script lang="ts">
	import { page } from '$app/state';
	import { base } from '$app/paths';
	import { navigation, type NavItem } from '$lib/docs/navigation';
	import { locale, type Locale } from '$lib/docs/i18n';
	import { Globe, Github, ArrowLeft, ChevronRight, Menu, X } from 'lucide-svelte';

	let { children } = $props();
	let mobileNavOpen = $state(false);

	let currentLang: Locale = $state('en');
	locale.subscribe(v => currentLang = v);

	function toggleLang() {
		const next: Locale = currentLang === 'en' ? 'ko' : 'en';
		locale.set(next);
		currentLang = next;
	}

	function getTitle(item: NavItem): string {
		return currentLang === 'ko' && item.titleKo ? item.titleKo : item.title;
	}

	let currentPath = $derived(page.url.pathname.replace(base, ''));

	let currentSection = $derived.by(() => {
		for (const section of navigation) {
			if (section.items) {
				for (const item of section.items) {
					if (currentPath === item.href || currentPath === item.href + '/') {
						return section;
					}
				}
			}
			if (currentPath === section.href || currentPath === section.href + '/') {
				return section;
			}
		}
		return undefined;
	});

	let breadcrumbs = $derived.by(() => {
		const crumbs: { title: string; href: string }[] = [];
		if (!currentSection) return crumbs;

		crumbs.push({ title: getTitle(currentSection), href: currentSection.href });

		if (currentSection.items) {
			const current = currentSection.items.find(
				i => currentPath === i.href || currentPath === i.href + '/'
			);
			if (current) {
				crumbs.push({ title: getTitle(current), href: current.href });
			}
		}
		return crumbs;
	});

	let isIndex = $derived(
		currentPath === '/docs' || currentPath === '/docs/'
	);

	const topSections = navigation.filter(s => s.items && s.items.length > 0);
</script>

<div class="vx-blog">
	<header class="vx-blog-header">
		<div class="vx-blog-header-inner">
			<div class="vx-blog-header-left">
				<a href="{base}/" class="vx-blog-logo">
					<span class="vx-blog-logo-v">V</span>
					<span class="vx-blog-logo-text">Vectrix</span>
				</a>
				<span class="vx-blog-logo-divider">/</span>
				<a href="{base}/docs" class="vx-blog-docs-link">Docs</a>
			</div>

			<nav class="vx-blog-tabs">
				{#each topSections as section}
					<a
						href="{base}{section.items?.[0]?.href ?? section.href}"
						class="vx-blog-tab"
						class:active={currentSection === section}
					>
						{getTitle(section)}
					</a>
				{/each}
				{#each navigation.filter(s => !s.items) as single}
					<a
						href="{base}{single.href}"
						class="vx-blog-tab"
						class:active={currentPath === single.href || currentPath === single.href + '/'}
					>
						{getTitle(single)}
					</a>
				{/each}
			</nav>

			<div class="vx-blog-header-right">
				<button class="vx-blog-lang-btn" onclick={toggleLang}>
					<Globe size={14} />
					{currentLang === 'en' ? 'EN' : 'KO'}
				</button>
				<a href="https://github.com/eddmpython/vectrix" target="_blank" rel="noopener" class="vx-blog-icon-link">
					<Github size={18} />
				</a>
				<button class="vx-blog-mobile-btn" onclick={() => mobileNavOpen = !mobileNavOpen}>
					{#if mobileNavOpen}<X size={20} />{:else}<Menu size={20} />{/if}
				</button>
			</div>
		</div>

		{#if mobileNavOpen}
			<nav class="vx-blog-mobile-nav">
				{#each topSections as section}
					<a
						href="{base}{section.items?.[0]?.href ?? section.href}"
						class="vx-blog-mobile-link"
						class:active={currentSection === section}
						onclick={() => mobileNavOpen = false}
					>
						{getTitle(section)}
					</a>
				{/each}
				{#each navigation.filter(s => !s.items) as single}
					<a
						href="{base}{single.href}"
						class="vx-blog-mobile-link"
						class:active={currentPath === single.href}
						onclick={() => mobileNavOpen = false}
					>
						{getTitle(single)}
					</a>
				{/each}
			</nav>
		{/if}
	</header>

	<main class="vx-blog-main">
		{#if !isIndex && breadcrumbs.length > 0}
			<div class="vx-blog-breadcrumb">
				<a href="{base}/docs">{currentLang === 'ko' ? '문서' : 'Docs'}</a>
				{#each breadcrumbs as crumb, i}
					<ChevronRight size={12} />
					{#if i < breadcrumbs.length - 1}
						<a href="{base}{crumb.href}">{crumb.title}</a>
					{:else}
						<span>{crumb.title}</span>
					{/if}
				{/each}
			</div>
		{/if}

		{#if !isIndex && currentSection?.items}
			<nav class="vx-blog-subnav">
				{#each currentSection.items as item}
					<a
						href="{base}{item.href}"
						class="vx-blog-subnav-link"
						class:active={currentPath === item.href || currentPath === item.href + '/'}
					>
						{getTitle(item)}
					</a>
				{/each}
			</nav>
		{/if}

		<div class="vx-blog-content">
			{@render children()}
		</div>
	</main>
</div>

<style>
	:global(body) {
		margin: 0;
		background: #0f172a;
		color: #f8fafc;
	}

	.vx-blog {
		min-height: 100vh;
	}

	/* Header */
	.vx-blog-header {
		position: sticky;
		top: 0;
		z-index: 50;
		background: rgba(2, 6, 23, 0.85);
		backdrop-filter: blur(12px);
		border-bottom: 1px solid rgba(148, 163, 184, 0.08);
	}

	.vx-blog-header-inner {
		max-width: 1100px;
		margin: 0 auto;
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 1.5rem;
		height: 56px;
	}

	.vx-blog-header-left {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.vx-blog-logo {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		text-decoration: none;
		color: #f8fafc;
		font-weight: 700;
		font-size: 1.05rem;
	}

	.vx-blog-logo-v {
		background: linear-gradient(135deg, #06b6d4, #8b5cf6);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		font-size: 1.3rem;
		font-weight: 900;
	}

	.vx-blog-logo-divider {
		color: #334155;
		font-size: 1.2rem;
		font-weight: 300;
		margin: 0 0.15rem;
	}

	.vx-blog-docs-link {
		color: #94a3b8;
		text-decoration: none;
		font-size: 0.9rem;
		font-weight: 500;
	}
	.vx-blog-docs-link:hover { color: #f8fafc; }

	/* Tabs */
	.vx-blog-tabs {
		display: flex;
		gap: 0;
	}

	.vx-blog-tab {
		padding: 0 0.75rem;
		height: 56px;
		display: flex;
		align-items: center;
		font-size: 0.82rem;
		font-weight: 500;
		color: #64748b;
		text-decoration: none;
		border-bottom: 2px solid transparent;
		transition: all 0.15s;
		white-space: nowrap;
	}

	.vx-blog-tab:hover { color: #cbd5e1; }
	.vx-blog-tab.active {
		color: #06b6d4;
		border-bottom-color: #06b6d4;
	}

	.vx-blog-header-right {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.vx-blog-lang-btn {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.3rem 0.6rem;
		border-radius: 6px;
		border: 1px solid rgba(148, 163, 184, 0.15);
		background: transparent;
		color: #94a3b8;
		font-size: 0.75rem;
		cursor: pointer;
		transition: all 0.15s;
	}
	.vx-blog-lang-btn:hover { border-color: #06b6d4; color: #06b6d4; }

	.vx-blog-icon-link {
		color: #64748b;
		display: flex;
		padding: 0.25rem;
		transition: color 0.15s;
	}
	.vx-blog-icon-link:hover { color: #f8fafc; }

	.vx-blog-mobile-btn {
		display: none;
		padding: 0.25rem;
		border: none;
		background: none;
		color: #94a3b8;
		cursor: pointer;
	}

	/* Main */
	.vx-blog-main {
		max-width: 860px;
		margin: 0 auto;
		padding: 1.5rem 1.5rem 4rem;
	}

	/* Breadcrumb */
	.vx-blog-breadcrumb {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.78rem;
		color: #475569;
		margin-bottom: 0.75rem;
	}

	.vx-blog-breadcrumb a {
		color: #64748b;
		text-decoration: none;
	}
	.vx-blog-breadcrumb a:hover { color: #06b6d4; }
	.vx-blog-breadcrumb span { color: #94a3b8; }

	/* Sub-navigation (section pages) */
	.vx-blog-subnav {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
		margin-bottom: 2rem;
		padding-bottom: 1rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.08);
	}

	.vx-blog-subnav-link {
		padding: 0.35rem 0.75rem;
		font-size: 0.8rem;
		color: #64748b;
		text-decoration: none;
		border-radius: 6px;
		transition: all 0.15s;
	}
	.vx-blog-subnav-link:hover {
		color: #cbd5e1;
		background: rgba(148, 163, 184, 0.06);
	}
	.vx-blog-subnav-link.active {
		color: #06b6d4;
		background: rgba(6, 182, 212, 0.08);
	}

	/* Content (markdown) */
	.vx-blog-content :global(h1) {
		font-size: 2rem;
		font-weight: 800;
		margin-bottom: 0.5rem;
		background: linear-gradient(135deg, #f8fafc, #94a3b8);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
	}

	.vx-blog-content :global(h2) {
		font-size: 1.5rem;
		font-weight: 700;
		margin-top: 2.5rem;
		margin-bottom: 0.75rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.1);
		color: #f8fafc;
	}

	.vx-blog-content :global(h3) {
		font-size: 1.2rem;
		font-weight: 600;
		margin-top: 2rem;
		margin-bottom: 0.5rem;
		color: #e2e8f0;
	}

	.vx-blog-content :global(h4) {
		font-size: 1rem;
		font-weight: 600;
		margin-top: 1.5rem;
		margin-bottom: 0.5rem;
		color: #cbd5e1;
	}

	.vx-blog-content :global(p) {
		line-height: 1.75;
		color: #94a3b8;
		margin-bottom: 1rem;
	}

	.vx-blog-content :global(a) {
		color: #06b6d4;
		text-decoration: none;
	}
	.vx-blog-content :global(a:hover) { text-decoration: underline; }

	.vx-blog-content :global(code:not(pre code)) {
		background: rgba(148, 163, 184, 0.1);
		padding: 0.15rem 0.4rem;
		border-radius: 4px;
		font-size: 0.875em;
		font-family: 'JetBrains Mono', monospace;
		color: #e2e8f0;
	}

	.vx-blog-content :global(pre) {
		background: #0d1117 !important;
		border: 1px solid rgba(148, 163, 184, 0.1);
		border-radius: 8px;
		padding: 1rem;
		overflow-x: auto;
		margin: 1rem 0;
		font-size: 0.85rem;
	}

	.vx-blog-content :global(pre code) {
		background: none !important;
		padding: 0;
		font-family: 'JetBrains Mono', monospace;
	}

	.vx-blog-content :global(ul), .vx-blog-content :global(ol) {
		padding-left: 1.5rem;
		margin-bottom: 1rem;
		color: #94a3b8;
	}

	.vx-blog-content :global(li) {
		line-height: 1.75;
		margin-bottom: 0.25rem;
	}

	.vx-blog-content :global(table) {
		width: 100%;
		border-collapse: collapse;
		margin: 1.5rem 0;
		font-size: 0.875rem;
	}

	.vx-blog-content :global(th) {
		text-align: left;
		padding: 0.75rem 1rem;
		border-bottom: 2px solid rgba(148, 163, 184, 0.2);
		color: #f8fafc;
		font-weight: 600;
		font-size: 0.8rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.vx-blog-content :global(td) {
		padding: 0.6rem 1rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.08);
		color: #94a3b8;
	}

	.vx-blog-content :global(tr:hover td) {
		background: rgba(148, 163, 184, 0.03);
	}

	.vx-blog-content :global(blockquote) {
		border-left: 3px solid #06b6d4;
		padding: 0.5rem 1rem;
		margin: 1rem 0;
		background: rgba(6, 182, 212, 0.05);
		border-radius: 0 6px 6px 0;
	}

	.vx-blog-content :global(blockquote p) {
		color: #cbd5e1;
		margin: 0;
	}

	.vx-blog-content :global(hr) {
		border: none;
		border-top: 1px solid rgba(148, 163, 184, 0.1);
		margin: 2rem 0;
	}

	.vx-blog-content :global(img) {
		max-width: 100%;
		border-radius: 8px;
	}

	/* Mobile nav */
	.vx-blog-mobile-nav {
		display: none;
		flex-direction: column;
		padding: 0.5rem 1rem 1rem;
		border-top: 1px solid rgba(148, 163, 184, 0.08);
	}

	.vx-blog-mobile-link {
		padding: 0.6rem 0.5rem;
		font-size: 0.85rem;
		color: #94a3b8;
		text-decoration: none;
		border-bottom: 1px solid rgba(148, 163, 184, 0.05);
	}
	.vx-blog-mobile-link:hover { color: #f8fafc; }
	.vx-blog-mobile-link.active { color: #06b6d4; }

	@media (max-width: 768px) {
		.vx-blog-tabs { display: none; }
		.vx-blog-mobile-btn { display: block; }
		.vx-blog-mobile-nav { display: flex; }

		.vx-blog-main { padding: 1rem 1rem 3rem; }

		.vx-blog-subnav {
			overflow-x: auto;
			flex-wrap: nowrap;
			-webkit-overflow-scrolling: touch;
		}
		.vx-blog-subnav-link { flex-shrink: 0; }
	}

	@media (max-width: 480px) {
		.vx-blog-logo-text { display: none; }
		.vx-blog-logo-divider { display: none; }
	}
</style>
