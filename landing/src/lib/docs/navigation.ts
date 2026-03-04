export interface NavItem {
	title: string;
	titleKo?: string;
	href: string;
	items?: NavItem[];
}

export const navigation: NavItem[] = [
	{
		title: 'Getting Started',
		titleKo: '시작하기',
		href: '/docs/getting-started',
		items: [
			{ title: 'Installation', titleKo: '설치', href: '/docs/getting-started/installation' },
			{ title: 'Quickstart', titleKo: '빠른 시작', href: '/docs/getting-started/quickstart' }
		]
	},
	{
		title: 'User Guide',
		titleKo: '사용자 가이드',
		href: '/docs/guide',
		items: [
			{ title: 'Forecasting', titleKo: '예측', href: '/docs/guide/forecasting' },
			{ title: 'Analysis & DNA', titleKo: '분석 & DNA', href: '/docs/guide/analysis' },
			{ title: 'Regression', titleKo: '회귀분석', href: '/docs/guide/regression' },
			{ title: 'Adaptive', titleKo: '적응형 예측', href: '/docs/guide/adaptive' },
			{ title: 'Business', titleKo: '비즈니스', href: '/docs/guide/business' },
			{ title: 'Pipeline', titleKo: '파이프라인', href: '/docs/guide/pipeline' },
			{ title: 'Foundation Models', titleKo: '기초 모델', href: '/docs/guide/foundation' },
			{ title: 'Multivariate', titleKo: '다변량', href: '/docs/guide/multivariate' }
		]
	},
	{
		title: 'API Reference',
		titleKo: 'API 레퍼런스',
		href: '/docs/api',
		items: [
			{ title: 'Easy API', titleKo: 'Easy API', href: '/docs/api/easy' },
			{ title: 'Vectrix Class', titleKo: 'Vectrix 클래스', href: '/docs/api/vectrix' },
			{ title: 'Adaptive', titleKo: '적응형', href: '/docs/api/adaptive' },
			{ title: 'Business', titleKo: '비즈니스', href: '/docs/api/business' },
			{ title: 'Regression', titleKo: '회귀분석', href: '/docs/api/regression' },
			{ title: 'Pipeline', titleKo: '파이프라인', href: '/docs/api/pipeline' },
			{ title: 'Foundation', titleKo: '기초 모델', href: '/docs/api/foundation' },
			{ title: 'Types', titleKo: '타입', href: '/docs/api/types' }
		]
	},
	{
		title: 'Tutorials',
		titleKo: '튜토리얼',
		href: '/docs/tutorials',
		items: [
			{ title: '1. Quickstart', titleKo: '1. 빠른 시작', href: '/docs/tutorials/quickstart' },
			{ title: '2. Analysis', titleKo: '2. 분석', href: '/docs/tutorials/analyze' },
			{ title: '3. Regression', titleKo: '3. 회귀분석', href: '/docs/tutorials/regression' },
			{ title: '4. Models', titleKo: '4. 모델 비교', href: '/docs/tutorials/models' },
			{ title: '5. Adaptive', titleKo: '5. 적응형', href: '/docs/tutorials/adaptive' },
			{ title: '6. Business', titleKo: '6. 비즈니스', href: '/docs/tutorials/business' }
		]
	},
	{
		title: 'Blog',
		titleKo: '블로그',
		href: '/blog'
	},
	{
		title: 'Benchmarks',
		titleKo: '벤치마크',
		href: '/docs/benchmarks'
	},
	{
		title: 'Changelog',
		titleKo: '변경 로그',
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
