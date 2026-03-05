export interface BlogPost {
	slug: string;
	title: string;
	description: string;
	category: 'fundamentals' | 'how-to' | 'deep-dive' | 'benchmarks';
	date: string;
	readingTime: string;
	featured?: boolean;
}

export const categories = {
	fundamentals: { label: 'Fundamentals', icon: 'school', color: '#06b6d4' },
	'how-to': { label: 'How-To', icon: 'code-braces', color: '#8b5cf6' },
	'deep-dive': { label: 'Deep Dive', icon: 'magnify', color: '#f59e0b' },
	benchmarks: { label: 'Benchmarks', icon: 'chart-bar', color: '#10b981' }
} as const;

export const blogPosts: BlogPost[] = [
	{
		slug: 'python-forecasting-libraries',
		title: 'Python Forecasting Libraries Compared — Which One Should You Use?',
		description: 'statsmodels, statsforecast, Prophet, Darts, sktime, NeuralForecast, and more — an honest comparison of every major Python forecasting library with benchmarks, code examples, and decision guide.',
		category: 'how-to',
		date: '2026-03-05',
		readingTime: '18 min',
		featured: true
	},
	{
		slug: 'how-we-know-forecasts-work',
		title: 'How Do We Know If a Forecast Is Any Good?',
		description: 'Benchmarks, metrics, and the M Competitions — the scientific method behind measuring forecast accuracy. From MAE to OWA, learn what separates real accuracy from marketing.',
		category: 'benchmarks',
		date: '2026-03-04',
		readingTime: '12 min',
		featured: true
	},
	{
		slug: 'what-is-forecasting',
		title: 'What Is Forecasting?',
		description: 'The one concept behind every business decision. Learn what forecasting really means, why it matters, and how it works — no math required.',
		category: 'fundamentals',
		date: '2026-03-04',
		readingTime: '8 min'
	}
];

export function getPostBySlug(slug: string): BlogPost | undefined {
	return blogPosts.find(p => p.slug === slug);
}

export function getPostsByCategory(category: string): BlogPost[] {
	return blogPosts.filter(p => p.category === category);
}
