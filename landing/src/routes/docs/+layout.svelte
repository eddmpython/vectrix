<script lang="ts">
	import { page } from '$app/state';
	import { base } from '$app/paths';
	import { navigation, findCurrentSection, findPrevNext, type NavItem } from '$lib/docs/navigation';
	import { locale, type Locale } from '$lib/docs/i18n';
	import { Book, ChevronRight, ChevronLeft, Globe, Menu, X, Search, Github, ArrowLeft } from 'lucide-svelte';

	let { children } = $props();
	let sidebarOpen = $state(false);
	let searchOpen = $state(false);
	let searchQuery = $state('');

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

	function isActive(href: string): boolean {
		const current = page.url.pathname.replace(base, '');
		return current === href || current === href + '/';
	}

	function isInSection(href: string): boolean {
		const current = page.url.pathname.replace(base, '');
		return current.startsWith(href + '/');
	}
</script>

<div class="vx-docs">
	<button class="vx-docs-menu-btn" onclick={() => sidebarOpen = !sidebarOpen}>
		{#if sidebarOpen}
			<X size={20} />
		{:else}
			<Menu size={20} />
		{/if}
	</button>

	<aside class="vx-docs-sidebar" class:open={sidebarOpen}>
		<div class="vx-docs-sidebar-header">
			<a href="{base}/" class="vx-docs-logo">
				<span class="vx-docs-logo-icon">V</span>
				<span class="vx-docs-logo-text">Vectrix</span>
			</a>
			<div class="vx-docs-sidebar-actions">
				<button class="vx-docs-lang-btn" onclick={toggleLang} title="Toggle language">
					<Globe size={16} />
					<span>{currentLang === 'en' ? 'EN' : 'KO'}</span>
				</button>
			</div>
		</div>

		<nav class="vx-docs-nav">
			{#each navigation as section}
				{#if section.items}
					<div class="vx-docs-nav-section">
						<span class="vx-docs-nav-label">{getTitle(section)}</span>
						{#each section.items as item}
							<a
								href="{base}{item.href}"
								class="vx-docs-nav-link"
								class:active={isActive(item.href)}
								onclick={() => sidebarOpen = false}
							>
								{getTitle(item)}
							</a>
						{/each}
					</div>
				{:else}
					<a
						href="{base}{section.href}"
						class="vx-docs-nav-link top-level"
						class:active={isActive(section.href)}
						onclick={() => sidebarOpen = false}
					>
						{getTitle(section)}
					</a>
				{/if}
			{/each}
		</nav>

		<div class="vx-docs-sidebar-footer">
			<a href="https://github.com/eddmpython/vectrix" target="_blank" rel="noopener" class="vx-docs-nav-link">
				<Github size={14} />
				GitHub
			</a>
			<a href="{base}/" class="vx-docs-nav-link">
				<ArrowLeft size={14} />
				{currentLang === 'ko' ? '홈으로' : 'Back to Home'}
			</a>
		</div>
	</aside>

	{#if sidebarOpen}
		<div class="vx-docs-overlay" onclick={() => sidebarOpen = false}></div>
	{/if}

	<main class="vx-docs-main">
		<div class="vx-docs-content">
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

	.vx-docs {
		display: flex;
		min-height: 100vh;
	}

	/* Sidebar */
	.vx-docs-sidebar {
		position: fixed;
		top: 0;
		left: 0;
		width: 280px;
		height: 100vh;
		background: #020617;
		border-right: 1px solid rgba(148, 163, 184, 0.1);
		display: flex;
		flex-direction: column;
		z-index: 40;
		overflow-y: auto;
	}

	.vx-docs-sidebar-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 1rem 1.25rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.1);
		position: sticky;
		top: 0;
		background: #020617;
		z-index: 1;
	}

	.vx-docs-logo {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		text-decoration: none;
		color: #f8fafc;
		font-weight: 700;
		font-size: 1.1rem;
	}

	.vx-docs-logo-icon {
		background: linear-gradient(135deg, #06b6d4, #8b5cf6);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		font-size: 1.4rem;
		font-weight: 900;
	}

	.vx-docs-sidebar-actions {
		display: flex;
		gap: 0.5rem;
	}

	.vx-docs-lang-btn {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.25rem 0.5rem;
		border-radius: 6px;
		border: 1px solid rgba(148, 163, 184, 0.2);
		background: transparent;
		color: #94a3b8;
		font-size: 0.75rem;
		cursor: pointer;
		transition: all 0.15s;
	}

	.vx-docs-lang-btn:hover {
		border-color: #06b6d4;
		color: #06b6d4;
	}

	/* Navigation */
	.vx-docs-nav {
		padding: 1rem 0;
		flex: 1;
	}

	.vx-docs-nav-section {
		margin-bottom: 0.5rem;
	}

	.vx-docs-nav-label {
		display: block;
		padding: 0.5rem 1.25rem 0.25rem;
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: #64748b;
	}

	.vx-docs-nav-link {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.4rem 1.25rem 0.4rem 1.75rem;
		font-size: 0.85rem;
		color: #94a3b8;
		text-decoration: none;
		transition: all 0.15s;
		border-left: 2px solid transparent;
	}

	.vx-docs-nav-link.top-level {
		padding-left: 1.25rem;
		font-weight: 500;
		margin-top: 0.25rem;
	}

	.vx-docs-nav-link:hover {
		color: #f8fafc;
		background: rgba(148, 163, 184, 0.05);
	}

	.vx-docs-nav-link.active {
		color: #06b6d4;
		border-left-color: #06b6d4;
		background: rgba(6, 182, 212, 0.05);
	}

	.vx-docs-sidebar-footer {
		padding: 1rem;
		border-top: 1px solid rgba(148, 163, 184, 0.1);
	}

	.vx-docs-sidebar-footer .vx-docs-nav-link {
		padding: 0.4rem 0.5rem;
		font-size: 0.8rem;
		border-left: none;
	}

	/* Main content */
	.vx-docs-main {
		flex: 1;
		margin-left: 280px;
		min-height: 100vh;
	}

	.vx-docs-content {
		max-width: 860px;
		margin: 0 auto;
		padding: 2.5rem 2rem;
	}

	/* Markdown content styling */
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

	.vx-docs-content :global(a) {
		color: #06b6d4;
		text-decoration: none;
	}

	.vx-docs-content :global(a:hover) {
		text-decoration: underline;
	}

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

	.vx-docs-content :global(blockquote p) {
		color: #cbd5e1;
		margin: 0;
	}

	.vx-docs-content :global(hr) {
		border: none;
		border-top: 1px solid rgba(148, 163, 184, 0.1);
		margin: 2rem 0;
	}

	.vx-docs-content :global(img) {
		max-width: 100%;
		border-radius: 8px;
	}

	/* Mobile */
	.vx-docs-menu-btn {
		display: none;
		position: fixed;
		top: 1rem;
		left: 1rem;
		z-index: 50;
		padding: 0.5rem;
		border-radius: 8px;
		border: 1px solid rgba(148, 163, 184, 0.2);
		background: #020617;
		color: #f8fafc;
		cursor: pointer;
	}

	.vx-docs-overlay {
		display: none;
	}

	@media (max-width: 768px) {
		.vx-docs-menu-btn {
			display: block;
		}

		.vx-docs-sidebar {
			transform: translateX(-100%);
			transition: transform 0.2s ease;
		}

		.vx-docs-sidebar.open {
			transform: translateX(0);
		}

		.vx-docs-overlay {
			display: block;
			position: fixed;
			inset: 0;
			background: rgba(0, 0, 0, 0.5);
			z-index: 30;
		}

		.vx-docs-main {
			margin-left: 0;
		}

		.vx-docs-content {
			padding: 3.5rem 1.25rem 2rem;
		}
	}
</style>
