<script lang="ts">
	import { base } from '$app/paths';
	import { getPostBySlug, blogPosts, categories } from '$lib/blog/posts';
	import { SITE_URL, SITE_NAME, OG_IMAGE } from '$lib/docs/seo';
	import { Clock, ArrowLeft, ArrowRight } from 'lucide-svelte';
	import { onMount, tick } from 'svelte';
	import Giscus from '$lib/components/ui/Giscus.svelte';

	let { data } = $props();

	let post = $derived(getPostBySlug(data.slug));
	let Component = $derived(data.component);

	let otherPosts = $derived(blogPosts.filter(p => p.slug !== data.slug));
	let visibleCount = $state(3);
	let visiblePosts = $derived(otherPosts.slice(0, visibleCount));
	let hasMore = $derived(visibleCount < otherPosts.length);
	let sentinel: HTMLDivElement | undefined = $state();

	let pageTitle = $derived(data.meta?.title ? `${data.meta.title} — Vectrix Blog` : 'Vectrix Blog');
	let pageDescription = $derived(post?.description ?? 'A blog post from Vectrix about forecasting.');
	let canonicalUrl = $derived(`${SITE_URL}/blog/${data.slug}`);
	let categoryInfo = $derived(post ? categories[post.category] : null);

	interface TocItem { id: string; text: string; level: number; }
	let tocItems: TocItem[] = $state([]);
	let activeId = $state('');
	let articleEl: HTMLElement | undefined = $state();
	let tocCleanup: (() => void) | undefined;
	let mounted = false;

	function extractToc() {
		if (!articleEl) return;
		const headings = articleEl.querySelectorAll('h2, h3');
		const items: TocItem[] = [];
		headings.forEach((h) => {
			if (!h.id) {
				h.id = h.textContent?.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '') ?? '';
			}
			items.push({ id: h.id, text: h.textContent?.trim() ?? '', level: h.tagName === 'H2' ? 2 : 3 });
		});
		tocItems = items;
	}

	function observeHeadings() {
		if (!articleEl) return;
		const headings = articleEl.querySelectorAll('h2, h3');
		if (headings.length === 0) return;
		const observer = new IntersectionObserver((entries) => {
			for (const entry of entries) {
				if (entry.isIntersecting) { activeId = entry.target.id; break; }
			}
		}, { rootMargin: '-80px 0px -70% 0px', threshold: 0 });
		headings.forEach(h => observer.observe(h));
		return () => observer.disconnect();
	}

	function scrollToHeading(id: string) {
		document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
	}

	function addCopyButtons() {
		document.querySelectorAll('.blog-article pre').forEach((pre) => {
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

	onMount(() => {
		mounted = true;
		return () => {
			mounted = false;
			tocCleanup?.();
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
			tocCleanup?.();
			tocCleanup = observeHeadings();
			if (tocItems.length === 0 && articleEl) {
				setTimeout(() => {
					extractToc();
					tocCleanup?.();
					tocCleanup = observeHeadings();
				}, 200);
			}
		});
	});

	onMount(() => {
		if (sentinel) {
			const observer = new IntersectionObserver((entries) => {
				if (entries[0].isIntersecting && hasMore) {
					visibleCount += 3;
				}
			}, { rootMargin: '200px' });
			observer.observe(sentinel);
			return () => observer.disconnect();
		}
	});
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
	{#if post}
		<meta property="article:published_time" content={post.date} />
		<meta property="article:section" content={categories[post.category].label} />
	{/if}

	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content={pageTitle} />
	<meta name="twitter:description" content={pageDescription} />
	<meta name="twitter:image" content={OG_IMAGE} />

	{@html `<script type="application/ld+json">${JSON.stringify({
		"@context": "https://schema.org",
		"@type": "BlogPosting",
		"headline": data.meta?.title ?? "Vectrix Blog",
		"description": pageDescription,
		"url": canonicalUrl,
		"image": OG_IMAGE,
		"datePublished": post?.date,
		"author": { "@type": "Person", "name": "eddmpython", "url": "https://github.com/eddmpython" },
		"publisher": { "@type": "Organization", "name": "Vectrix", "logo": { "@type": "ImageObject", "url": OG_IMAGE } },
		"mainEntityOfPage": { "@type": "WebPage", "@id": canonicalUrl },
		"inLanguage": "en"
	})}</script>`}

	{@html `<script type="application/ld+json">${JSON.stringify({
		"@context": "https://schema.org",
		"@type": "BreadcrumbList",
		"itemListElement": [
			{ "@type": "ListItem", "position": 1, "name": "Blog", "item": `${SITE_URL}/blog` },
			{ "@type": "ListItem", "position": 2, "name": data.meta?.title ?? data.slug, "item": canonicalUrl }
		]
	})}</script>`}
</svelte:head>

{#if data.status === 404}
	<div class="not-found">
		<h1>404</h1>
		<p>Post not found: <code>{data.error}</code></p>
		<a href="{base}/blog">Back to Blog</a>
	</div>
{:else}
	<div class="blog-post-wrapper">
		<div class="blog-post-page">
			<a href="{base}/blog" class="blog-back">
				<ArrowLeft size={14} />
				All Posts
			</a>

			{#if post}
				<div class="blog-post-meta">
					<span class="blog-post-category" style="color: {categoryInfo?.color}">{categoryInfo?.label}</span>
					<span class="blog-post-date">{post.date}</span>
					<span class="blog-post-reading">
						<Clock size={12} />
						{post.readingTime}
					</span>
				</div>
			{/if}

			<article class="blog-article" bind:this={articleEl}>
				<Component />
			</article>

			<div class="blog-post-footer">
				<a href="{base}/blog" class="blog-back-bottom">
					<ArrowLeft size={14} />
					Back to all posts
				</a>
			</div>

			<Giscus />

		{#if otherPosts.length > 0}
			<section class="more-posts">
				<h2 class="more-posts-heading">More from the blog</h2>
				{#each visiblePosts as p}
					<a href="{base}/blog/{p.slug}" class="more-post-card">
						<div class="more-post-meta">
							<span class="more-post-category" style="color: {categories[p.category].color}">{categories[p.category].label}</span>
							<span class="more-post-date">{p.date}</span>
							<span class="more-post-reading">
								<Clock size={12} />
								{p.readingTime}
							</span>
						</div>
						<h3 class="more-post-title">{p.title}</h3>
						<p class="more-post-desc">{p.description}</p>
						<span class="more-post-cta">
							Read article <ArrowRight size={14} />
						</span>
					</a>
				{/each}

				{#if hasMore}
					<div bind:this={sentinel} class="more-posts-sentinel">
						<div class="more-posts-loader"></div>
					</div>
				{/if}
			</section>
		{/if}
		</div>

		{#if tocItems.length > 0}
			<aside class="blog-toc">
				<div class="blog-toc-inner">
					<span class="blog-toc-heading">On this page</span>
					<nav class="blog-toc-list">
						{#each tocItems as item}
							<button
								class="blog-toc-item"
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

	.blog-post-wrapper {
		position: relative;
		max-width: 720px;
		margin: 0 auto;
	}

	.blog-post-page {
		min-width: 0;
	}

	.blog-back {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		font-size: 0.82rem;
		color: #64748b;
		text-decoration: none;
		margin-bottom: 1.5rem;
		transition: color 0.15s;
	}
	.blog-back:hover { color: #06b6d4; }

	.blog-post-meta {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 0.5rem;
	}

	.blog-post-category {
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.blog-post-date {
		font-size: 0.75rem;
		color: #475569;
	}

	.blog-post-reading {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.75rem;
		color: #475569;
	}

	.blog-article :global(h1) {
		font-size: 2.2rem;
		font-weight: 800;
		margin: 0.25rem 0 1rem;
		background: linear-gradient(135deg, #f8fafc, #94a3b8);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		line-height: 1.2;
	}

	.blog-article :global(h2) {
		font-size: 1.5rem;
		font-weight: 700;
		margin-top: 3.5rem;
		margin-bottom: 1rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.1);
		color: #f8fafc;
	}

	.blog-article :global(h3) {
		font-size: 1.2rem;
		font-weight: 600;
		margin-top: 2.5rem;
		margin-bottom: 0.75rem;
		color: #e2e8f0;
	}

	.blog-article :global(p) {
		line-height: 1.8;
		color: #94a3b8;
		margin-bottom: 1.25rem;
		font-size: 1rem;
	}

	.blog-article :global(a) {
		color: #06b6d4;
		text-decoration: none;
	}
	.blog-article :global(a:hover) { text-decoration: underline; }

	.blog-article :global(strong) {
		color: #e2e8f0;
		font-weight: 600;
	}

	.blog-article :global(code:not(pre code)) {
		background: rgba(148, 163, 184, 0.1);
		padding: 0.15rem 0.4rem;
		border-radius: 4px;
		font-size: 0.875em;
		font-family: 'JetBrains Mono', monospace;
		color: #e2e8f0;
	}

	.blog-article :global(pre) {
		background: #0d1117 !important;
		border: 1px solid rgba(148, 163, 184, 0.1);
		border-radius: 8px;
		padding: 1rem;
		overflow-x: auto;
		margin: 1.25rem 0;
		font-size: 0.85rem;
	}

	.blog-article :global(pre code) {
		background: none !important;
		padding: 0;
		font-family: 'JetBrains Mono', monospace;
	}

	.blog-article :global(ul), .blog-article :global(ol) {
		padding-left: 1.5rem;
		margin-bottom: 1.25rem;
		color: #94a3b8;
	}

	.blog-article :global(li) {
		line-height: 1.8;
		margin-bottom: 0.35rem;
	}

	.blog-article :global(blockquote) {
		border-left: 3px solid #06b6d4;
		padding: 0.75rem 1.25rem;
		margin: 1.5rem 0;
		background: rgba(6, 182, 212, 0.05);
		border-radius: 0 6px 6px 0;
	}

	.blog-article :global(blockquote p) {
		color: #cbd5e1;
		margin: 0;
	}

	.blog-article :global(table) {
		width: 100%;
		border-collapse: collapse;
		margin: 1.5rem 0;
		font-size: 0.875rem;
	}

	.blog-article :global(th) {
		text-align: left;
		padding: 0.75rem 1rem;
		border-bottom: 2px solid rgba(148, 163, 184, 0.2);
		color: #f8fafc;
		font-weight: 600;
		font-size: 0.8rem;
	}

	.blog-article :global(td) {
		padding: 0.6rem 1rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.08);
		color: #94a3b8;
	}

	.blog-article :global(hr) {
		border: none;
		border-top: 1px solid rgba(148, 163, 184, 0.1);
		margin: 2rem 0;
	}

	.blog-article :global(img) {
		max-width: 100%;
		width: 100%;
		height: auto;
		border-radius: 8px;
		margin: 1.5rem 0;
	}

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

	.blog-post-footer {
		margin-top: 3rem;
		padding-top: 1.5rem;
		border-top: 1px solid rgba(148, 163, 184, 0.1);
	}

	.blog-back-bottom {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		font-size: 0.85rem;
		color: #64748b;
		text-decoration: none;
		transition: color 0.15s;
	}
	.blog-back-bottom:hover { color: #06b6d4; }

	/* --- More posts (infinite scroll) --- */

	.more-posts {
		margin-top: 4rem;
		padding-top: 2rem;
		border-top: 1px solid rgba(148, 163, 184, 0.08);
	}

	.more-posts-heading {
		font-size: 0.8rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: #64748b;
		margin: 0 0 1.25rem;
	}

	.more-post-card {
		display: block;
		padding: 1.25rem;
		border-radius: 10px;
		border: 1px solid rgba(148, 163, 184, 0.08);
		text-decoration: none;
		transition: all 0.2s;
		margin-bottom: 0.75rem;
	}

	.more-post-card:hover {
		border-color: rgba(6, 182, 212, 0.3);
		background: rgba(6, 182, 212, 0.03);
	}

	.more-post-meta {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 0.4rem;
	}

	.more-post-category {
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.more-post-date {
		font-size: 0.7rem;
		color: #475569;
	}

	.more-post-reading {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.7rem;
		color: #475569;
	}

	.more-post-title {
		font-size: 1.15rem;
		font-weight: 700;
		color: #f8fafc;
		margin: 0 0 0.3rem;
		line-height: 1.3;
	}

	.more-post-desc {
		font-size: 0.85rem;
		color: #94a3b8;
		line-height: 1.6;
		margin: 0;
	}

	.more-post-cta {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		margin-top: 0.6rem;
		font-size: 0.8rem;
		font-weight: 500;
		color: #06b6d4;
	}

	.more-posts-sentinel {
		display: flex;
		justify-content: center;
		padding: 2rem 0;
	}

	.more-posts-loader {
		width: 24px;
		height: 24px;
		border: 2px solid rgba(148, 163, 184, 0.15);
		border-top-color: #06b6d4;
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	/* Blog TOC */
	.blog-toc {
		position: fixed;
		top: 72px;
		left: calc(50% + 360px + 2rem);
		width: 200px;
		height: fit-content;
		max-height: calc(100vh - 90px);
		overflow-y: auto;
		scrollbar-width: thin;
		scrollbar-color: rgba(148, 163, 184, 0.15) transparent;
	}

	.blog-toc-inner {
		padding-top: 0.5rem;
	}

	.blog-toc-heading {
		display: block;
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: #475569;
		margin-bottom: 0.6rem;
	}

	.blog-toc-list {
		display: flex;
		flex-direction: column;
	}

	.blog-toc-item {
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

	.blog-toc-item:hover { color: #cbd5e1; }

	.blog-toc-item.active {
		color: #06b6d4;
		border-left-color: #06b6d4;
	}

	.blog-toc-item.h3 {
		padding-left: 1.1rem;
		font-size: 0.72rem;
	}

	@media (max-width: 1100px) {
		.blog-toc { display: none; }
	}

	@media (max-width: 480px) {
		.blog-article :global(h1) { font-size: 1.6rem; }
	}
</style>
