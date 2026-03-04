<script lang="ts">
	import { base } from '$app/paths';
	import { blogPosts, categories } from '$lib/blog/posts';
	import { SITE_URL, SITE_NAME, OG_IMAGE } from '$lib/docs/seo';
	import { BookOpen, Clock, ArrowRight } from 'lucide-svelte';

	const title = 'Blog — Forecasting, Explained | Vectrix';
	const description = 'Learn forecasting from zero. Concepts, tutorials, benchmarks, and best practices for prediction with Python.';
	const canonicalUrl = `${SITE_URL}/blog`;

	const featured = blogPosts.filter(p => p.featured);
	const recent = blogPosts.slice(0, 10);
</script>

<svelte:head>
	<title>{title}</title>
	<meta name="description" content={description} />
	<link rel="canonical" href={canonicalUrl} />

	<meta property="og:type" content="website" />
	<meta property="og:title" content={title} />
	<meta property="og:description" content={description} />
	<meta property="og:url" content={canonicalUrl} />
	<meta property="og:image" content={OG_IMAGE} />
	<meta property="og:site_name" content={SITE_NAME} />

	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content={title} />
	<meta name="twitter:description" content={description} />
	<meta name="twitter:image" content={OG_IMAGE} />

	{@html `<script type="application/ld+json">${JSON.stringify({
		"@context": "https://schema.org",
		"@type": "Blog",
		"name": "Vectrix Blog",
		"description": description,
		"url": canonicalUrl,
		"publisher": {
			"@type": "Organization",
			"name": "Vectrix",
			"url": SITE_URL
		},
		"blogPost": blogPosts.map(p => ({
			"@type": "BlogPosting",
			"headline": p.title,
			"description": p.description,
			"datePublished": p.date,
			"url": `${SITE_URL}/blog/${p.slug}`,
			"author": { "@type": "Person", "name": "eddmpython" }
		}))
	})}</script>`}
</svelte:head>

<div class="blog-index">
	<div class="blog-hero">
		<h1>Forecasting, Explained</h1>
		<p class="blog-hero-sub">From fundamental concepts to advanced techniques — everything you need to master forecasting.</p>
		<p class="blog-hero-audience">Whether you're a business analyst predicting next quarter's sales, a data scientist building ML pipelines, or a student learning statistics — this blog is for you.</p>
	</div>

	<div class="blog-categories">
		{#each Object.entries(categories) as [key, cat]}
			<div class="blog-category-card">
				<div class="blog-category-dot" style="background: {cat.color}"></div>
				<div>
					<span class="blog-category-label">{cat.label}</span>
					<span class="blog-category-count">{blogPosts.filter(p => p.category === key).length}</span>
				</div>
			</div>
		{/each}
	</div>

	{#if featured.length > 0}
		<section class="blog-section">
			<h2>Featured</h2>
			{#each featured as post}
				<a href="{base}/blog/{post.slug}" class="blog-card featured">
					<div class="blog-card-meta">
						<span class="blog-card-category" style="color: {categories[post.category].color}">{categories[post.category].label}</span>
						<span class="blog-card-date">{post.date}</span>
						<span class="blog-card-reading">
							<Clock size={12} />
							{post.readingTime}
						</span>
					</div>
					<h3 class="blog-card-title">{post.title}</h3>
					<p class="blog-card-desc">{post.description}</p>
					<span class="blog-card-cta">
						Read article <ArrowRight size={14} />
					</span>
				</a>
			{/each}
		</section>
	{/if}

	<section class="blog-section">
		<h2>All Posts</h2>
		{#each recent as post}
			<a href="{base}/blog/{post.slug}" class="blog-card">
				<div class="blog-card-meta">
					<span class="blog-card-category" style="color: {categories[post.category].color}">{categories[post.category].label}</span>
					<span class="blog-card-date">{post.date}</span>
					<span class="blog-card-reading">
						<Clock size={12} />
						{post.readingTime}
					</span>
				</div>
				<h3 class="blog-card-title">{post.title}</h3>
				<p class="blog-card-desc">{post.description}</p>
			</a>
		{/each}
	</section>
</div>

<style>
	.blog-index {
		max-width: 720px;
		margin: 0 auto;
	}

	.blog-hero {
		margin-bottom: 3rem;
	}

	.blog-hero h1 {
		font-size: 2.5rem;
		font-weight: 800;
		margin: 0 0 0.75rem;
		background: linear-gradient(135deg, #f8fafc, #94a3b8);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		line-height: 1.2;
	}

	.blog-hero-sub {
		font-size: 1.1rem;
		color: #94a3b8;
		line-height: 1.6;
		margin: 0 0 0.75rem;
	}

	.blog-hero-audience {
		font-size: 0.9rem;
		color: #64748b;
		line-height: 1.6;
		margin: 0;
	}

	.blog-categories {
		display: flex;
		gap: 0.75rem;
		flex-wrap: wrap;
		margin-bottom: 3rem;
	}

	.blog-category-card {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.85rem;
		border-radius: 8px;
		border: 1px solid rgba(148, 163, 184, 0.1);
		background: rgba(148, 163, 184, 0.03);
	}

	.blog-category-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}

	.blog-category-label {
		font-size: 0.8rem;
		color: #cbd5e1;
		font-weight: 500;
	}

	.blog-category-count {
		font-size: 0.7rem;
		color: #475569;
		margin-left: 0.25rem;
	}

	.blog-section {
		margin-bottom: 2.5rem;
	}

	.blog-section h2 {
		font-size: 0.8rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: #64748b;
		margin: 0 0 1rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.08);
	}

	.blog-card {
		display: block;
		padding: 1.25rem;
		border-radius: 10px;
		border: 1px solid rgba(148, 163, 184, 0.08);
		text-decoration: none;
		transition: all 0.2s;
		margin-bottom: 0.75rem;
	}

	.blog-card:hover {
		border-color: rgba(6, 182, 212, 0.3);
		background: rgba(6, 182, 212, 0.03);
	}

	.blog-card.featured {
		border-color: rgba(6, 182, 212, 0.15);
		background: rgba(6, 182, 212, 0.02);
	}

	.blog-card-meta {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 0.5rem;
	}

	.blog-card-category {
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.blog-card-date {
		font-size: 0.75rem;
		color: #475569;
	}

	.blog-card-reading {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.75rem;
		color: #475569;
	}

	.blog-card-title {
		font-size: 1.25rem;
		font-weight: 700;
		color: #f8fafc;
		margin: 0 0 0.35rem;
		line-height: 1.3;
	}

	.blog-card-desc {
		font-size: 0.88rem;
		color: #94a3b8;
		line-height: 1.6;
		margin: 0;
	}

	.blog-card-cta {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		margin-top: 0.75rem;
		font-size: 0.82rem;
		font-weight: 500;
		color: #06b6d4;
	}

	@media (max-width: 480px) {
		.blog-hero h1 { font-size: 1.8rem; }
		.blog-categories { gap: 0.5rem; }
	}
</style>
