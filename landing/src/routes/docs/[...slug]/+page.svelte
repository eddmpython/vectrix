<script lang="ts">
	import { locale } from '$lib/docs/i18n';
	import { base } from '$app/paths';
	import { navigation, findPrevNext } from '$lib/docs/navigation';
	import { getDescription, getCanonicalUrl, SITE_NAME, OG_IMAGE, SITE_URL } from '$lib/docs/seo';
	import { page } from '$app/state';
	import { ChevronLeft, ChevronRight } from 'lucide-svelte';
	import { onMount, tick } from 'svelte';

	let { data } = $props();
	let currentLang = $state('en');
	locale.subscribe(v => currentLang = v);

	interface TocItem {
		id: string;
		text: string;
		level: number;
	}

	let tocItems: TocItem[] = $state([]);
	let activeId = $state('');
	let articleEl: HTMLElement | undefined = $state();

	function extractToc() {
		if (!articleEl) return;
		const headings = articleEl.querySelectorAll('h2, h3');
		const items: TocItem[] = [];
		headings.forEach((h) => {
			if (!h.id) {
				h.id = h.textContent?.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '') ?? '';
			}
			items.push({
				id: h.id,
				text: h.textContent?.trim() ?? '',
				level: h.tagName === 'H2' ? 2 : 3
			});
		});
		tocItems = items;
	}

	function observeHeadings() {
		if (!articleEl) return;
		const headings = articleEl.querySelectorAll('h2, h3');
		if (headings.length === 0) return;

		const observer = new IntersectionObserver((entries) => {
			for (const entry of entries) {
				if (entry.isIntersecting) {
					activeId = entry.target.id;
					break;
				}
			}
		}, { rootMargin: '-80px 0px -70% 0px', threshold: 0 });

		headings.forEach(h => observer.observe(h));
		return () => observer.disconnect();
	}

	function addCopyButtons() {
		document.querySelectorAll('.doc-article pre').forEach((pre) => {
			if (pre.querySelector('.copy-btn')) return;
			const wrapper = document.createElement('div');
			wrapper.style.position = 'relative';
			pre.parentNode!.insertBefore(wrapper, pre);
			wrapper.appendChild(pre);

			const btn = document.createElement('button');
			btn.className = 'copy-btn';
			btn.textContent = 'Copy';
			btn.addEventListener('click', () => {
				const code = pre.querySelector('code');
				const text = (code || pre).textContent || '';
				navigator.clipboard.writeText(text).then(() => {
					btn.textContent = 'Copied!';
					setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
				});
			});
			wrapper.appendChild(btn);
		});
	}

	let cleanup: (() => void) | undefined;
	let mounted = false;

	onMount(() => {
		mounted = true;
		return () => {
			mounted = false;
			cleanup?.();
		};
	});

	$effect(() => {
		if (!mounted) return;
		Component;
		data;
		tick().then(() => {
			if (!mounted) return;
			addCopyButtons();
			extractToc();
			cleanup?.();
			cleanup = observeHeadings();
			if (tocItems.length === 0 && articleEl) {
				setTimeout(() => {
					extractToc();
					cleanup?.();
					cleanup = observeHeadings();
				}, 200);
			}
		});
	});

	let Component = $derived(
		currentLang === 'ko' && data.koComponent ? data.koComponent : data.enComponent
	);
	let meta = $derived(
		currentLang === 'ko' && data.koMeta?.title ? data.koMeta : (data.enMeta ?? {})
	);

	let prevNext = $derived(findPrevNext(page.url.pathname.replace(base, ''), navigation));

	let slugPath = $derived(`/docs/${data.slug}`);
	let pageTitle = $derived(meta?.title ? `${meta.title} — Vectrix` : 'Vectrix Docs');
	let pageDescription = $derived(getDescription(slugPath));
	let canonicalUrl = $derived(getCanonicalUrl(slugPath));

	let breadcrumbItems = $derived(() => {
		if (!data.slug) return [{ name: 'Docs', url: `${SITE_URL}/docs` }];
		const parts = data.slug.split('/');
		const items = [{ name: 'Docs', url: `${SITE_URL}/docs` }];
		let path = '/docs';
		for (const part of parts) {
			path += `/${part}`;
			const name = part.replace(/-/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase());
			items.push({ name, url: `${SITE_URL}${path}` });
		}
		return items;
	});

	function scrollToHeading(id: string) {
		const el = document.getElementById(id);
		if (el) {
			el.scrollIntoView({ behavior: 'smooth', block: 'start' });
		}
	}
</script>

<svelte:head>
	<title>{pageTitle}</title>
	<meta name="description" content={pageDescription} />
	<link rel="canonical" href={canonicalUrl} />

	<meta property="og:type" content="article" />
	<meta property="og:title" content={pageTitle} />
	<meta property="og:description" content={pageDescription} />
	<meta property="og:url" content={canonicalUrl} />
	<meta property="og:image" content={OG_IMAGE} />
	<meta property="og:site_name" content={SITE_NAME} />

	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content={pageTitle} />
	<meta name="twitter:description" content={pageDescription} />
	<meta name="twitter:image" content={OG_IMAGE} />

	{@html `<script type="application/ld+json">${JSON.stringify({
		"@context": "https://schema.org",
		"@type": "TechArticle",
		"headline": meta?.title || "Vectrix Documentation",
		"description": pageDescription,
		"url": canonicalUrl,
		"image": OG_IMAGE,
		"author": { "@type": "Person", "name": "eddmpython", "url": "https://github.com/eddmpython" },
		"publisher": { "@type": "Organization", "name": "Vectrix", "logo": { "@type": "ImageObject", "url": OG_IMAGE } },
		"mainEntityOfPage": { "@type": "WebPage", "@id": canonicalUrl },
		"inLanguage": currentLang === "ko" ? "ko" : "en"
	})}</script>`}

	{@html `<script type="application/ld+json">${JSON.stringify({
		"@context": "https://schema.org",
		"@type": "BreadcrumbList",
		"itemListElement": breadcrumbItems().map((item, i) => ({
			"@type": "ListItem",
			"position": i + 1,
			"name": item.name,
			"item": item.url
		}))
	})}</script>`}
</svelte:head>

{#if data.status === 404}
	<div class="not-found">
		<h1>404</h1>
		<p>Page not found: <code>{data.error}</code></p>
		<a href="{base}/docs/getting-started/installation">Go to Installation</a>
	</div>
{:else}
	<div class="doc-layout">
		<div class="doc-content-col">
			<article class="doc-article" bind:this={articleEl}>
				<Component />
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
		</div>

		{#if tocItems.length > 0}
			<aside class="doc-toc">
				<div class="doc-toc-inner">
					<span class="doc-toc-heading">On this page</span>
					<nav class="doc-toc-list">
						{#each tocItems as item}
							<button
								class="doc-toc-item"
								class:h3={item.level === 3}
								class:active={activeId === item.id}
								onclick={() => scrollToHeading(item.id)}
							>
								{item.text}
							</button>
						{/each}
					</nav>
				</div>
			</aside>
		{/if}
	</div>
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
	.not-found p { color: #94a3b8; margin: 1rem 0; }
	.not-found a { color: #06b6d4; text-decoration: none; }

	/* 2-column: content + toc */
	.doc-layout {
		display: grid;
		grid-template-columns: 1fr 200px;
		gap: 2rem;
	}

	.doc-content-col {
		min-width: 0;
	}

	/* Right TOC */
	.doc-toc {
		position: sticky;
		top: 72px;
		height: fit-content;
		max-height: calc(100vh - 90px);
		overflow-y: auto;
		scrollbar-width: thin;
		scrollbar-color: rgba(148, 163, 184, 0.15) transparent;
	}

	.doc-toc-inner {
		padding-top: 0.5rem;
	}

	.doc-toc-heading {
		display: block;
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: #475569;
		margin-bottom: 0.6rem;
	}

	.doc-toc-list {
		display: flex;
		flex-direction: column;
	}

	.doc-toc-item {
		display: block;
		width: 100%;
		text-align: left;
		padding: 0.2rem 0 0.2rem 0.6rem;
		font-size: 0.75rem;
		color: #64748b;
		background: none;
		border: none;
		border-left: 2px solid transparent;
		cursor: pointer;
		transition: all 0.12s;
		line-height: 1.4;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.doc-toc-item:hover {
		color: #cbd5e1;
	}

	.doc-toc-item.active {
		color: #06b6d4;
		border-left-color: #06b6d4;
	}

	.doc-toc-item.h3 {
		padding-left: 1.1rem;
		font-size: 0.72rem;
	}

	/* Copy buttons */
	:global(.copy-btn) {
		position: absolute;
		top: 8px;
		right: 8px;
		padding: 4px 10px;
		font-size: 0.7rem;
		font-family: 'JetBrains Mono', monospace;
		background: rgba(148, 163, 184, 0.15);
		color: #94a3b8;
		border: 1px solid rgba(148, 163, 184, 0.2);
		border-radius: 4px;
		cursor: pointer;
		opacity: 0;
		transition: opacity 0.15s, background 0.15s;
		z-index: 1;
	}
	:global(.copy-btn:hover) {
		background: rgba(6, 182, 212, 0.2);
		color: #06b6d4;
		border-color: rgba(6, 182, 212, 0.4);
	}
	:global(div:hover > .copy-btn) {
		opacity: 1;
	}

	/* Pagination */
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

	@media (max-width: 1200px) {
		.doc-layout {
			grid-template-columns: 1fr;
		}
		.doc-toc {
			display: none;
		}
	}
</style>
