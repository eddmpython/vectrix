export interface BlogPost {
	slug: string;
	title: string;
	description: string;
	category: 'fundamentals' | 'how-to' | 'deep-dive' | 'benchmarks';
	date: string;
	readingTime: string;
	featured?: boolean;
	keywords?: string[];
}

export const categories = {
	fundamentals: { label: 'Fundamentals', icon: 'school', color: '#06b6d4' },
	'how-to': { label: 'How-To', icon: 'code-braces', color: '#8b5cf6' },
	'deep-dive': { label: 'Deep Dive', icon: 'magnify', color: '#f59e0b' },
	benchmarks: { label: 'Benchmarks', icon: 'chart-bar', color: '#10b981' }
} as const;

export const blogPosts: BlogPost[] = [
	{
		slug: 'forecasting-models-explained',
		title: 'Forecasting Models Explained — From ETS and ARIMA to Foundation Models',
		description: 'A deep dive into 22 statistical models and the new wave of foundation models. ETS, ARIMA, Theta, GARCH, Croston — then Chronos-2, TimesFM, Moirai, and more. When each shines, when each struggles.',
		category: 'deep-dive',
		date: '2026-03-05',
		readingTime: '28 min',
		featured: true,
		keywords: ['ETS vs ARIMA', 'theta forecasting', 'foundation models time series', 'Chronos-2', 'TimesFM', 'Moirai', 'forecasting models comparison']
	},
	{
		slug: 'your-first-forecast-in-python',
		title: 'Your First Forecast in Python — Step by Step',
		description: 'From zero to prediction in 15 minutes. Load data, analyze patterns, generate forecasts with confidence intervals, compare 30+ models, and export results — all in Python.',
		category: 'how-to',
		date: '2026-03-05',
		readingTime: '15 min',
		featured: true,
		keywords: ['python forecasting tutorial', 'time series prediction python', 'forecast python', 'vectrix tutorial', 'first forecast']
	},
	{
		slug: 'python-forecasting-libraries',
		title: 'Python Forecasting Libraries Compared — Which One Should You Use?',
		description: 'statsmodels, statsforecast, Prophet, Darts, sktime, NeuralForecast, and more — an honest comparison of every major Python forecasting library with benchmarks, code examples, and decision guide.',
		category: 'how-to',
		date: '2026-03-05',
		readingTime: '18 min',
		featured: true,
		keywords: ['python forecasting library', 'statsforecast vs prophet', 'time series python', 'forecasting library comparison', 'darts vs sktime']
	},
	{
		slug: 'how-we-know-forecasts-work',
		title: 'How Do We Know If a Forecast Is Any Good?',
		description: 'Benchmarks, metrics, and the M Competitions — the scientific method behind measuring forecast accuracy. From MAE to OWA, learn what separates real accuracy from marketing.',
		category: 'benchmarks',
		date: '2026-03-04',
		readingTime: '12 min',
		featured: true,
		keywords: ['forecast accuracy', 'M competition', 'MAPE RMSE explained', 'forecast benchmark', 'OWA metric']
	},
	{
		slug: 'what-is-forecasting',
		title: 'What Is Forecasting?',
		description: 'The one concept behind every business decision. Learn what forecasting really means, why it matters, and how it works — no math required.',
		category: 'fundamentals',
		date: '2026-03-04',
		readingTime: '8 min',
		keywords: ['what is forecasting', 'forecasting basics', 'time series explained', 'demand forecasting', 'prediction vs forecasting']
	}
];

export function getPostBySlug(slug: string): BlogPost | undefined {
	return blogPosts.find(p => p.slug === slug);
}

export function getPostsByCategory(category: string): BlogPost[] {
	return blogPosts.filter(p => p.category === category);
}
