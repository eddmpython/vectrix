<script lang="ts">
	import { page } from '$app/state';
	import { base } from '$app/paths';
	import { navigation, type NavItem } from '$lib/docs/navigation';
	import { Github, Menu, X, ChevronRight, ChevronDown } from 'lucide-svelte';

	let { children } = $props();
	let mobileNavOpen = $state(false);

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

	let isIndex = $derived(
		currentPath === '/docs' || currentPath === '/docs/'
	);

	const sidebarSections = navigation.filter(s => s.items && s.items.length > 0);
	const standaloneItems = navigation.filter(s => !s.items || s.items.length === 0);

	let expandedSections = $state<Set<string>>(new Set());

	$effect(() => {
		if (currentSection) {
			expandedSections = new Set([currentSection.href]);
		}
	});

	function toggleSection(href: string) {
		const next = new Set(expandedSections);
		if (next.has(href)) {
			next.delete(href);
		} else {
			next.add(href);
		}
		expandedSections = next;
	}
</script>

<div class="vx-docs">
	<header class="vx-docs-header">
		<div class="vx-docs-header-inner">
			<div class="vx-docs-header-left">
				<a href="{base}/" class="vx-docs-logo">
					<img src="{base}/icon-final.png" alt="Vectrix" width="24" height="24" class="vx-docs-logo-img" />
					<span class="vx-docs-logo-text">Vectrix</span>
				</a>
				<span class="vx-docs-divider">/</span>
				<a href="{base}/docs" class="vx-docs-link">Docs</a>
			</div>

			<div class="vx-docs-header-right">
				<a href="https://github.com/eddmpython/vectrix" target="_blank" rel="noopener" class="vx-docs-icon-link">
					<Github size={18} />
				</a>
				<button class="vx-docs-mobile-btn" onclick={() => mobileNavOpen = !mobileNavOpen}>
					{#if mobileNavOpen}<X size={20} />{:else}<Menu size={20} />{/if}
				</button>
			</div>
		</div>

		{#if mobileNavOpen}
			<nav class="vx-docs-mobile-nav">
				{#each sidebarSections as section}
					<div class="vx-docs-mobile-section">
						<span class="vx-docs-mobile-section-title">{section.title}</span>
						{#each section.items ?? [] as item}
							<a
								href="{base}{item.href}"
								class="vx-docs-mobile-link"
								class:active={currentPath === item.href || currentPath === item.href + '/'}
								onclick={() => mobileNavOpen = false}
							>
								{item.title}
							</a>
						{/each}
					</div>
				{/each}
				{#each standaloneItems as item}
					<a
						href="{base}{item.href}"
						class="vx-docs-mobile-link"
						class:active={currentPath === item.href}
						onclick={() => mobileNavOpen = false}
					>
						{item.title}
					</a>
				{/each}
			</nav>
		{/if}
	</header>

	<div class="vx-docs-body" class:is-index={isIndex}>
		{#if !isIndex}
			<aside class="vx-docs-sidebar">
				<nav class="vx-docs-sidebar-nav">
					{#each sidebarSections as section}
						<div class="vx-sidebar-section">
							<button
								class="vx-sidebar-section-btn"
								class:active={currentSection === section}
								onclick={() => toggleSection(section.href)}
							>
								<span>{section.title}</span>
								{#if expandedSections.has(section.href)}
									<ChevronDown size={14} />
								{:else}
									<ChevronRight size={14} />
								{/if}
							</button>
							{#if expandedSections.has(section.href) && section.items}
								<div class="vx-sidebar-items">
									{#each section.items as item}
										<a
											href="{base}{item.href}"
											class="vx-sidebar-item"
											class:active={currentPath === item.href || currentPath === item.href + '/'}
										>
											{item.title}
										</a>
									{/each}
								</div>
							{/if}
						</div>
					{/each}
					<div class="vx-sidebar-standalone">
						{#each standaloneItems as item}
							<a
								href="{base}{item.href}"
								class="vx-sidebar-item"
								class:active={currentPath === item.href || currentPath === item.href + '/'}
							>
								{item.title}
							</a>
						{/each}
					</div>
				</nav>
			</aside>
		{/if}

		<main class="vx-docs-main">
			<div class="vx-docs-content">
				{@render children()}
			</div>
		</main>
	</div>
</div>

<style>
	:global(body) {
		margin: 0;
		background: #0f172a;
		color: #f8fafc;
	}

	.vx-docs {
		min-height: 100vh;
	}

	/* Header */
	.vx-docs-header {
		position: sticky;
		top: 0;
		z-index: 50;
		background: rgba(2, 6, 23, 0.85);
		backdrop-filter: blur(12px);
		border-bottom: 1px solid rgba(148, 163, 184, 0.08);
	}

	.vx-docs-header-inner {
		max-width: 1280px;
		margin: 0 auto;
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 1.5rem;
		height: 56px;
	}

	.vx-docs-header-left {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.vx-docs-logo {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		text-decoration: none;
		color: #f8fafc;
		font-weight: 700;
		font-size: 1.05rem;
	}

	.vx-docs-logo-img { border-radius: 4px; }

	.vx-docs-divider {
		color: #334155;
		font-size: 1.2rem;
		font-weight: 300;
		margin: 0 0.15rem;
	}

	.vx-docs-link {
		color: #94a3b8;
		text-decoration: none;
		font-size: 0.9rem;
		font-weight: 500;
	}
	.vx-docs-link:hover { color: #f8fafc; }

	.vx-docs-header-right {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.vx-docs-icon-link {
		color: #64748b;
		display: flex;
		padding: 0.25rem;
		transition: color 0.15s;
	}
	.vx-docs-icon-link:hover { color: #f8fafc; }

	.vx-docs-mobile-btn {
		display: none;
		padding: 0.25rem;
		border: none;
		background: none;
		color: #94a3b8;
		cursor: pointer;
	}

	/* Body 3-column */
	.vx-docs-body {
		max-width: 1280px;
		margin: 0 auto;
		display: grid;
		grid-template-columns: 220px 1fr;
		gap: 0;
		min-height: calc(100vh - 56px);
	}

	.vx-docs-body.is-index {
		grid-template-columns: 1fr;
	}

	/* Sidebar */
	.vx-docs-sidebar {
		position: sticky;
		top: 56px;
		height: calc(100vh - 56px);
		overflow-y: auto;
		padding: 1.25rem 0 2rem 1rem;
		border-right: 1px solid rgba(148, 163, 184, 0.06);
		scrollbar-width: thin;
		scrollbar-color: rgba(148, 163, 184, 0.15) transparent;
	}

	.vx-sidebar-section {
		margin-bottom: 0.25rem;
	}

	.vx-sidebar-section-btn {
		display: flex;
		align-items: center;
		justify-content: space-between;
		width: 100%;
		padding: 0.4rem 0.6rem;
		border: none;
		background: none;
		color: #94a3b8;
		font-size: 0.78rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		cursor: pointer;
		border-radius: 6px;
		transition: all 0.15s;
	}
	.vx-sidebar-section-btn:hover {
		color: #cbd5e1;
		background: rgba(148, 163, 184, 0.05);
	}
	.vx-sidebar-section-btn.active {
		color: #06b6d4;
	}

	.vx-sidebar-items {
		padding: 0.15rem 0 0.5rem 0.5rem;
	}

	.vx-sidebar-item {
		display: block;
		padding: 0.3rem 0.6rem;
		font-size: 0.8rem;
		color: #64748b;
		text-decoration: none;
		border-radius: 5px;
		transition: all 0.12s;
		border-left: 2px solid transparent;
	}
	.vx-sidebar-item:hover {
		color: #cbd5e1;
		background: rgba(148, 163, 184, 0.05);
	}
	.vx-sidebar-item.active {
		color: #06b6d4;
		background: rgba(6, 182, 212, 0.06);
		border-left-color: #06b6d4;
	}

	.vx-sidebar-standalone {
		margin-top: 0.75rem;
		padding-top: 0.75rem;
		border-top: 1px solid rgba(148, 163, 184, 0.06);
	}

	/* Main content */
	.vx-docs-main {
		padding: 1.5rem 2rem 4rem;
		min-width: 0;
	}

	/* Content styles */
	.vx-docs-content :global(h1) {
		font-size: 2rem;
		font-weight: 800;
		margin-bottom: 0.5rem;
		background: linear-gradient(135deg, #f8fafc, #94a3b8);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
	}

	.vx-docs-content :global(h2) {
		font-size: 1.5rem;
		font-weight: 700;
		margin-top: 2.5rem;
		margin-bottom: 0.75rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.1);
		color: #f8fafc;
	}

	.vx-docs-content :global(h3) {
		font-size: 1.2rem;
		font-weight: 600;
		margin-top: 2rem;
		margin-bottom: 0.5rem;
		color: #e2e8f0;
	}

	.vx-docs-content :global(h4) {
		font-size: 1rem;
		font-weight: 600;
		margin-top: 1.5rem;
		margin-bottom: 0.5rem;
		color: #cbd5e1;
	}

	.vx-docs-content :global(p) {
		line-height: 1.75;
		color: #94a3b8;
		margin-bottom: 1rem;
	}

	.vx-docs-content :global(a) { color: #06b6d4; text-decoration: none; }
	.vx-docs-content :global(a:hover) { text-decoration: underline; }

	.vx-docs-content :global(strong) { color: #e2e8f0; }

	.vx-docs-content :global(code:not(pre code)) {
		background: rgba(148, 163, 184, 0.1);
		padding: 0.15rem 0.4rem;
		border-radius: 4px;
		font-size: 0.875em;
		font-family: 'JetBrains Mono', monospace;
		color: #e2e8f0;
	}

	.vx-docs-content :global(pre) {
		background: #0d1117 !important;
		border: 1px solid rgba(148, 163, 184, 0.1);
		border-radius: 8px;
		padding: 1rem;
		overflow-x: auto;
		margin: 1rem 0;
		font-size: 0.85rem;
	}

	.vx-docs-content :global(pre code) {
		background: none !important;
		padding: 0;
		font-family: 'JetBrains Mono', monospace;
	}

	.vx-docs-content :global(ul), .vx-docs-content :global(ol) {
		padding-left: 1.5rem;
		margin-bottom: 1rem;
		color: #94a3b8;
	}

	.vx-docs-content :global(li) {
		line-height: 1.75;
		margin-bottom: 0.25rem;
	}

	.vx-docs-content :global(table) {
		width: 100%;
		border-collapse: collapse;
		margin: 1.5rem 0;
		font-size: 0.875rem;
	}

	.vx-docs-content :global(th) {
		text-align: left;
		padding: 0.75rem 1rem;
		border-bottom: 2px solid rgba(148, 163, 184, 0.2);
		color: #f8fafc;
		font-weight: 600;
		font-size: 0.8rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.vx-docs-content :global(td) {
		padding: 0.6rem 1rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.08);
		color: #94a3b8;
	}

	.vx-docs-content :global(tr:hover td) {
		background: rgba(148, 163, 184, 0.03);
	}

	.vx-docs-content :global(blockquote) {
		border-left: 3px solid #06b6d4;
		padding: 0.5rem 1rem;
		margin: 1rem 0;
		background: rgba(6, 182, 212, 0.05);
		border-radius: 0 6px 6px 0;
	}

	.vx-docs-content :global(blockquote p) { color: #cbd5e1; margin: 0; }

	.vx-docs-content :global(hr) {
		border: none;
		border-top: 1px solid rgba(148, 163, 184, 0.1);
		margin: 2rem 0;
	}

	.vx-docs-content :global(img) { max-width: 100%; border-radius: 8px; }

	/* Mobile nav */
	.vx-docs-mobile-nav {
		display: none;
		flex-direction: column;
		padding: 0.5rem 1rem 1rem;
		border-top: 1px solid rgba(148, 163, 184, 0.08);
		max-height: 60vh;
		overflow-y: auto;
	}

	.vx-docs-mobile-section {
		margin-bottom: 0.5rem;
	}

	.vx-docs-mobile-section-title {
		display: block;
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: #475569;
		padding: 0.4rem 0.5rem 0.15rem;
	}

	.vx-docs-mobile-link {
		display: block;
		padding: 0.5rem 0.5rem 0.5rem 1rem;
		font-size: 0.85rem;
		color: #94a3b8;
		text-decoration: none;
		border-bottom: 1px solid rgba(148, 163, 184, 0.05);
	}
	.vx-docs-mobile-link:hover { color: #f8fafc; }
	.vx-docs-mobile-link.active { color: #06b6d4; }

	@media (max-width: 1024px) {
		.vx-docs-body {
			grid-template-columns: 1fr;
		}

		.vx-docs-sidebar {
			display: none;
		}

		.vx-docs-mobile-btn { display: block; }
		.vx-docs-mobile-nav { display: flex; }

		.vx-docs-main { padding: 1rem 1rem 3rem; }
	}

	@media (max-width: 480px) {
		.vx-docs-logo-text { display: none; }
		.vx-docs-divider { display: none; }
	}
</style>
