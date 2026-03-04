import type { PageLoad } from './$types';
import type { EntryGenerator } from './$types';
import { blogPosts } from '$lib/blog/posts';

const modules = import.meta.glob('@docs/blog/*.md', { eager: true });

function stripNumberPrefix(filename: string): string {
	return filename.replace(/\/\d+_/g, '/');
}

const slugMap = new Map<string, string>();
for (const rawPath of Object.keys(modules)) {
	const slug = stripNumberPrefix(rawPath)
		.replace(/^.*\/blog\//, '')
		.replace('.md', '');
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
