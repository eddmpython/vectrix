import type { PageLoad } from './$types';
import type { EntryGenerator } from './$types';
import { blogPosts } from '$lib/blog/posts';

const modules = import.meta.glob('@docs/blog/*.md', { eager: true });

function toKebabSlug(filename: string): string {
	return filename
		.replace(/\/\d+_/g, '/')
		.replace(/^.*\/blog\//, '')
		.replace('.md', '')
		.replace(/([a-z])([A-Z])/g, '$1-$2')
		.toLowerCase();
}

const slugMap = new Map<string, string>();
for (const rawPath of Object.keys(modules)) {
	const slug = toKebabSlug(rawPath);
	slugMap.set(slug, rawPath);
}

export const entries: EntryGenerator = () => {
	return blogPosts.map((p) => ({ slug: p.slug }));
};

export const prerender = true;

export const load: PageLoad = async ({ params }) => {
	const slug = params.slug;
	const rawPath = slugMap.get(slug);

	if (!rawPath) {
		return {
			status: 404,
			error: `Post not found: ${slug}`
		};
	}

	const module = modules[rawPath] as any;

	return {
		component: module.default,
		meta: module.metadata ?? {},
		slug
	};
};
