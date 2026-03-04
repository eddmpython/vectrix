<script lang="ts">
	import { base } from '$app/paths';
	import { Github, Coffee } from 'lucide-svelte';
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();
</script>

<div class="vx-blog-shell">
	<header class="vx-blog-header">
		<div class="vx-blog-header-inner">
			<div class="vx-blog-header-left">
				<a href="{base}/" class="vx-blog-logo">
					<img src="{base}/icon-final.png" alt="Vectrix" width="24" height="24" class="vx-blog-logo-img" />
					<span class="vx-blog-logo-text">Vectrix</span>
				</a>
				<span class="vx-blog-logo-divider">/</span>
				<a href="{base}/blog" class="vx-blog-nav-link">Blog</a>
			</div>

			<nav class="vx-blog-tabs">
				<a href="{base}/docs/getting-started/installation" class="vx-blog-tab">Docs</a>
				<a href="{base}/blog" class="vx-blog-tab active">Blog</a>
			</nav>

			<div class="vx-blog-header-right">
				<a href="https://github.com/eddmpython/vectrix" target="_blank" rel="noopener" class="vx-blog-icon-link">
					<Github size={18} />
				</a>
			</div>
		</div>
	</header>

	<main class="vx-blog-main">
		{@render children()}
	</main>

	<footer class="vx-blog-footer">
		<div class="vx-blog-footer-inner">
			<span>&copy; {new Date().getFullYear()} Vectrix</span>
			<span class="vx-blog-footer-sep">&middot;</span>
			<a href="{base}/docs/getting-started/installation">Docs</a>
			<span class="vx-blog-footer-sep">&middot;</span>
			<a href="https://github.com/eddmpython/vectrix">GitHub</a>
			<span class="vx-blog-footer-sep">&middot;</span>
			<a href="https://pypi.org/project/vectrix/">PyPI</a>
			<span class="vx-blog-footer-sep">&middot;</span>
			<a href="https://buymeacoffee.com/eddmpython" target="_blank" rel="noopener" class="vx-blog-coffee-link">
				<Coffee size={13} />
				Buy me a coffee
			</a>
		</div>
	</footer>
</div>

<style>
	:global(body) {
		margin: 0;
		background: #0f172a;
		color: #f8fafc;
	}

	.vx-blog-shell {
		min-height: 100vh;
		display: flex;
		flex-direction: column;
	}

	.vx-blog-header {
		position: sticky;
		top: 0;
		z-index: 50;
		background: rgba(2, 6, 23, 0.85);
		backdrop-filter: blur(12px);
		border-bottom: 1px solid rgba(148, 163, 184, 0.08);
	}

	.vx-blog-header-inner {
		max-width: 860px;
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

	.vx-blog-logo-img {
		border-radius: 4px;
	}

	.vx-blog-logo-divider {
		color: #334155;
		font-size: 1.2rem;
		font-weight: 300;
		margin: 0 0.15rem;
	}

	.vx-blog-nav-link {
		color: #06b6d4;
		text-decoration: none;
		font-size: 0.9rem;
		font-weight: 500;
	}

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

	.vx-blog-icon-link {
		color: #64748b;
		display: flex;
		padding: 0.25rem;
		transition: color 0.15s;
	}
	.vx-blog-icon-link:hover { color: #f8fafc; }

	.vx-blog-main {
		flex: 1;
		max-width: 860px;
		width: 100%;
		margin: 0 auto;
		padding: 2rem 1.5rem 4rem;
	}

	.vx-blog-footer {
		border-top: 1px solid rgba(148, 163, 184, 0.08);
		padding: 1.5rem;
	}

	.vx-blog-footer-inner {
		max-width: 860px;
		margin: 0 auto;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.78rem;
		color: #475569;
	}

	.vx-blog-footer a {
		color: #64748b;
		text-decoration: none;
	}
	.vx-blog-footer a:hover { color: #06b6d4; }

	:global(.vx-blog-coffee-link) {
		display: inline-flex !important;
		align-items: center;
		gap: 0.3rem;
	}

	.vx-blog-footer-sep {
		color: #334155;
	}

	@media (max-width: 768px) {
		.vx-blog-tabs { display: none; }
		.vx-blog-main { padding: 1.5rem 1rem 3rem; }
	}

	@media (max-width: 480px) {
		.vx-blog-logo-text { display: none; }
		.vx-blog-logo-divider { display: none; }
	}
</style>
