import type { PageLoad } from './$types';
import type { EntryGenerator } from './$types';

const modules = import.meta.glob('/src/content/**/*.md', { eager: true });

const enSlugs = Object.keys(modules)
	.filter(p => p.startsWith('/src/content/en/'))
	.map(p => p.replace('/src/content/en/', '').replace('.md', ''));

export const entries: EntryGenerator = () => {
	return enSlugs.map(slug => ({ slug }));
};

export const prerender = true;

export const load: PageLoad = async ({ params }) => {
	const slug = params.slug;

	const enPath = `/src/content/en/${slug}.md`;
	const koPath = `/src/content/ko/${slug}.md`;

	const enModule = modules[enPath] as any;
	const koModule = modules[koPath] as any;

	if (!enModule) {
		return {
			status: 404,
			error: `Page not found: ${slug}`
		};
	}

	return {
		enComponent: enModule.default,
		koComponent: koModule?.default ?? null,
		enMeta: enModule.metadata ?? {},
		koMeta: koModule?.metadata ?? {},
		slug
	};
};
