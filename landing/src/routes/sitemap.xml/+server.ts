import type { RequestHandler } from './$types';

const SITE = 'https://eddmpython.github.io/vectrix';

const pages = [
	'/',
	'/docs',
	'/docs/getting-started/installation',
	'/docs/getting-started/quickstart',
	'/docs/guide/forecasting',
	'/docs/guide/analysis',
	'/docs/guide/regression',
	'/docs/guide/adaptive',
	'/docs/guide/business',
	'/docs/guide/pipeline',
	'/docs/guide/foundation',
	'/docs/guide/multivariate',
	'/docs/api/easy',
	'/docs/api/vectrix',
	'/docs/api/adaptive',
	'/docs/api/business',
	'/docs/api/regression',
	'/docs/api/pipeline',
	'/docs/api/foundation',
	'/docs/api/types',
	'/docs/tutorials/quickstart',
	'/docs/tutorials/analyze',
	'/docs/tutorials/regression',
	'/docs/tutorials/models',
	'/docs/tutorials/adaptive',
	'/docs/tutorials/business',
	'/blog',
	'/blog/what-is-forecasting',
	'/docs/benchmarks',
	'/docs/changelog'
];

export const prerender = true;

export const GET: RequestHandler = () => {
	const lastmod = new Date().toISOString().split('T')[0];

	const urls = pages
		.map(
			(p) => `  <url>
    <loc>${SITE}${p}</loc>
    <lastmod>${lastmod}</lastmod>
    <changefreq>${p === '/' || p.startsWith('/blog') ? 'weekly' : 'monthly'}</changefreq>
    <priority>${p === '/' ? '1.0' : p.startsWith('/docs/getting-started') ? '0.9' : p.startsWith('/blog') ? '0.8' : '0.7'}</priority>
  </url>`
		)
		.join('\n');

	const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls}
</urlset>`;

	return new Response(xml, {
		headers: {
			'Content-Type': 'application/xml',
			'Cache-Control': 'max-age=0, s-maxage=3600'
		}
	});
};
