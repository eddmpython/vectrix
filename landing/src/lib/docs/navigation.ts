export interface NavItem {
	title: string;
	href: string;
	items?: NavItem[];
}

export const navigation: NavItem[] = [
	{
		title: 'Getting Started',
		href: '/docs/getting-started',
		items: [
			{ title: 'Installation', href: '/docs/getting-started/installation' },
			{ title: 'Quickstart', href: '/docs/getting-started/quickstart' }
		]
	},
	{
		title: 'User Guide',
		href: '/docs/guide',
		items: [
			{ title: 'Forecasting', href: '/docs/guide/forecasting' },
			{ title: 'Analysis & DNA', href: '/docs/guide/analysis' },
			{ title: 'Regression', href: '/docs/guide/regression' },
			{ title: 'Adaptive', href: '/docs/guide/adaptive' },
			{ title: 'Business', href: '/docs/guide/business' },
			{ title: 'Pipeline', href: '/docs/guide/pipeline' },
			{ title: 'Foundation Models', href: '/docs/guide/foundation' },
			{ title: 'Multivariate', href: '/docs/guide/multivariate' }
		]
	},
	{
		title: 'API Reference',
		href: '/docs/api',
		items: [
			{ title: 'Easy API', href: '/docs/api/easy' },
			{ title: 'Vectrix Class', href: '/docs/api/vectrix' },
			{ title: 'Adaptive', href: '/docs/api/adaptive' },
			{ title: 'Business', href: '/docs/api/business' },
			{ title: 'Regression', href: '/docs/api/regression' },
			{ title: 'Pipeline', href: '/docs/api/pipeline' },
			{ title: 'Foundation', href: '/docs/api/foundation' },
			{ title: 'Types', href: '/docs/api/types' }
		]
	},
	{
		title: 'Tutorials',
		href: '/docs/tutorials',
		items: [
			{ title: '1. Quickstart', href: '/docs/tutorials/quickstart' },
			{ title: '2. Analysis', href: '/docs/tutorials/analyze' },
			{ title: '3. Regression', href: '/docs/tutorials/regression' },
			{ title: '4. Models', href: '/docs/tutorials/models' },
			{ title: '5. Adaptive', href: '/docs/tutorials/adaptive' },
			{ title: '6. Business', href: '/docs/tutorials/business' }
		]
	},
	{
		title: 'Blog',
		href: '/blog'
	},
	{
		title: 'Benchmarks',
		href: '/docs/benchmarks'
	},
	{
		title: 'Changelog',
		href: '/docs/changelog'
	}
];

export function flattenNav(items: NavItem[]): NavItem[] {
	const flat: NavItem[] = [];
	for (const item of items) {
		flat.push(item);
		if (item.items) {
			flat.push(...flattenNav(item.items));
		}
	}
	return flat;
}

export function findCurrentSection(path: string, items: NavItem[]): NavItem | undefined {
	const flat = flattenNav(items);
	return flat.find(item => path.endsWith(item.href) || path.endsWith(item.href + '/'));
}

export function findPrevNext(path: string, items: NavItem[]): { prev?: NavItem; next?: NavItem } {
	const flat = flattenNav(items).filter(i => !i.items || i.items.length === 0);
	const idx = flat.findIndex(item => path.endsWith(item.href) || path.endsWith(item.href + '/'));
	return {
		prev: idx > 0 ? flat[idx - 1] : undefined,
		next: idx < flat.length - 1 ? flat[idx + 1] : undefined
	};
}
