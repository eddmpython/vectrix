<script lang="ts">
	import { locale } from '$lib/docs/i18n';
	import { base } from '$app/paths';
	import { navigation, findPrevNext } from '$lib/docs/navigation';
	import { page } from '$app/state';
	import { ChevronLeft, ChevronRight } from 'lucide-svelte';

	let { data } = $props();
	let currentLang = $state('en');
	locale.subscribe(v => currentLang = v);

	let component = $derived(
		currentLang === 'ko' && data.koComponent ? data.koComponent : data.enComponent
	);
	let meta = $derived(
		currentLang === 'ko' && data.koMeta?.title ? data.koMeta : data.enMeta
	);

	let prevNext = $derived(findPrevNext(page.url.pathname.replace(base, ''), navigation));
</script>

<svelte:head>
	<title>{meta.title ? `${meta.title} — Vectrix` : 'Vectrix Docs'}</title>
</svelte:head>

{#if data.status === 404}
	<div class="not-found">
		<h1>404</h1>
		<p>Page not found: <code>{data.error}</code></p>
		<a href="{base}/docs/getting-started/installation">Go to Installation →</a>
	</div>
{:else}
	<article class="doc-article">
		<component />
	</article>

	{#if prevNext.prev || prevNext.next}
		<nav class="doc-pagination">
			{#if prevNext.prev}
				<a href="{base}{prevNext.prev.href}" class="doc-pagination-link prev">
					<ChevronLeft size={16} />
					<div>
						<span class="doc-pagination-label">Previous</span>
						<span class="doc-pagination-title">{currentLang === 'ko' && prevNext.prev.titleKo ? prevNext.prev.titleKo : prevNext.prev.title}</span>
					</div>
				</a>
			{:else}
				<div></div>
			{/if}
			{#if prevNext.next}
				<a href="{base}{prevNext.next.href}" class="doc-pagination-link next">
					<div>
						<span class="doc-pagination-label">Next</span>
						<span class="doc-pagination-title">{currentLang === 'ko' && prevNext.next.titleKo ? prevNext.next.titleKo : prevNext.next.title}</span>
					</div>
					<ChevronRight size={16} />
				</a>
			{/if}
		</nav>
	{/if}
{/if}

<style>
	.not-found {
		text-align: center;
		padding: 4rem 2rem;
	}
	.not-found h1 {
		font-size: 4rem;
		font-weight: 800;
		background: linear-gradient(135deg, #06b6d4, #8b5cf6);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
	}
	.not-found p {
		color: #94a3b8;
		margin: 1rem 0;
	}
	.not-found a {
		color: #06b6d4;
		text-decoration: none;
	}

	.doc-pagination {
		display: flex;
		justify-content: space-between;
		gap: 1rem;
		margin-top: 3rem;
		padding-top: 1.5rem;
		border-top: 1px solid rgba(148, 163, 184, 0.1);
	}

	.doc-pagination-link {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.75rem 1rem;
		border-radius: 8px;
		border: 1px solid rgba(148, 163, 184, 0.1);
		text-decoration: none;
		color: #94a3b8;
		transition: all 0.15s;
		max-width: 45%;
	}

	.doc-pagination-link:hover {
		border-color: #06b6d4;
		color: #06b6d4;
	}

	.doc-pagination-link.next {
		margin-left: auto;
		text-align: right;
	}

	.doc-pagination-label {
		display: block;
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: #64748b;
	}

	.doc-pagination-title {
		display: block;
		font-size: 0.9rem;
		font-weight: 500;
	}
</style>
