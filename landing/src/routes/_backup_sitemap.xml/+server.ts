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
	'/docs/tutorials/01_quickstart/',
	'/docs/tutorials/02_analyze/',
	'/docs/tutorials/03_regression/',
	'/docs/tutorials/04_models/',
	'/docs/tutorials/05_adaptive/',
	'/docs/tutorials/06_business/',
	'/docs/tutorials/07_visualization/',
	'/playground',
	'/blog',
	'/blog/what-is-forecasting',
	'/blog/how-we-know-forecasts-work',
	'/blog/python-forecasting-libraries',
	'/blog/your-first-forecast-in-python',
	'/blog/forecasting-models-explained',
	'/blog/statistical-vs-foundation',
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
    <changefreq>${p === '/' || p.startsWith('/blog') || p === '/playground' ? 'weekly' : 'monthly'}</changefreq>
    <priority>${p === '/' ? '1.0' : p === '/playground' ? '0.9' : p.startsWith('/docs/getting-started') ? '0.9' : p.startsWith('/blog') ? '0.8' : '0.7'}</priority>
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
