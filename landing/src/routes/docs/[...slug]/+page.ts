import type { PageLoad } from './$types';
import type { EntryGenerator } from './$types';

const enModules = import.meta.glob(
	[
		'@docs/getting-started/*.md',
		'@docs/guide/*.md',
		'@docs/api/*.md',
		'@docs/tutorials/*.md',
		'@docs/benchmarks.md',
		'@docs/changelog.md',
		'!@docs/**/*.ko.md'
	],
	{ eager: true }
);

function stripNumberPrefix(filename: string): string {
	return filename.replace(/\/\d+_/g, '/');
}

function normalizePath(rawPath: string): string {
	return stripNumberPrefix(rawPath).replace(/^.*?\/docs\//, '');
}

const slugMap = new Map<string, string>();
const enSlugs: string[] = [];

for (const rawPath of Object.keys(enModules)) {
	if (rawPath.endsWith('/index.md')) continue;

	const slug = normalizePath(rawPath).replace('.md', '');
	slugMap.set(slug, rawPath);
	enSlugs.push(slug);
}

export const entries: EntryGenerator = () => {
	return enSlugs.map((slug) => ({ slug }));
};

export const prerender = true;

export const load: PageLoad = async ({ params }) => {
	const slug = params.slug;

	const enRawPath = slugMap.get(slug);
	if (!enRawPath) {
		return {
			status: 404,
			error: `Page not found: ${slug}`
		};
	}

	const enModule = enModules[enRawPath] as any;

	return {
		enComponent: enModule.default,
		enMeta: enModule.metadata ?? {},
		slug
	};
};
