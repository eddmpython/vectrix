<script lang="ts">
	import { base } from '$app/paths';
	import { getPostBySlug, categories } from '$lib/blog/posts';
	import { SITE_URL, SITE_NAME, OG_IMAGE } from '$lib/docs/seo';
	import { ChevronLeft, Clock, ArrowLeft } from 'lucide-svelte';
	import { onMount, tick } from 'svelte';

	let { data } = $props();

	let post = $derived(getPostBySlug(data.slug));
	let Component = $derived(data.component);

	let pageTitle = $derived(data.meta?.title ? `${data.meta.title} — Vectrix Blog` : 'Vectrix Blog');
	let pageDescription = $derived(post?.description ?? 'A blog post from Vectrix about forecasting.');
	let canonicalUrl = $derived(`${SITE_URL}/blog/${data.slug}`);
	let categoryInfo = $derived(post ? categories[post.category] : null);

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

	onMount(addCopyButtons);

	$effect(() => {
		Component;
		tick().then(addCopyButtons);
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

		<article class="blog-article">
			<Component />
		</article>

		<div class="blog-post-footer">
			<a href="{base}/blog" class="blog-back-bottom">
				<ArrowLeft size={14} />
				Back to all posts
			</a>
		</div>
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

	.blog-post-page {
		max-width: 720px;
		margin: 0 auto;
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
		margin-top: 2.5rem;
		margin-bottom: 0.75rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.1);
		color: #f8fafc;
	}

	.blog-article :global(h3) {
		font-size: 1.2rem;
		font-weight: 600;
		margin-top: 2rem;
		margin-bottom: 0.5rem;
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
		border-radius: 8px;
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

	@media (max-width: 480px) {
		.blog-article :global(h1) { font-size: 1.6rem; }
	}
</style>
