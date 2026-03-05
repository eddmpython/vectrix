<script lang="ts">
	import { base } from '$app/paths';
	import { Github, Youtube, Mail } from 'lucide-svelte';

	const footerLinks = {
		Documentation: [
			{ label: 'Installation', href: `${base}/docs/getting-started/installation` },
			{ label: 'Quickstart', href: `${base}/docs/getting-started/quickstart` },
			{ label: 'API Reference', href: `${base}/docs/api/easy` },
			{ label: 'User Guide', href: `${base}/docs/guide/forecasting` },
			{ label: 'Playground', href: `${base}/playground` }
		],
		Project: [
			{ label: 'GitHub', href: 'https://github.com/eddmpython/vectrix' },
			{ label: 'PyPI', href: 'https://pypi.org/project/vectrix/' },
			{ label: 'Changelog', href: `${base}/docs/changelog` },
			{ label: 'Benchmarks', href: `${base}/docs/benchmarks` },
			{ label: 'Issues', href: 'https://github.com/eddmpython/vectrix/issues' }
		],
		eddmpython: [
			{ label: 'GitHub Profile', href: 'https://github.com/eddmpython' },
			{ label: 'PyPI Profile', href: 'https://pypi.org/user/eddmpython/' },
			{ label: 'tradix', href: 'https://github.com/eddmpython/tradix' },
			{ label: 'eddmPrint', href: 'https://github.com/eddmpython/eddmPrint' }
		],
		Connect: [
			{ label: 'YouTube', href: 'https://www.youtube.com/@eddmpython' },
			{ label: 'Threads', href: 'https://www.threads.net/@eddmpython' },
			{ label: 'Tistory Blog', href: 'https://eddm.tistory.com' },
			{ label: 'Email', href: 'mailto:eddmpython@gmail.com' }
		]
	};

	const socialLinks = [
		{ icon: Github, href: 'https://github.com/eddmpython', label: 'GitHub' },
		{ icon: Youtube, href: 'https://www.youtube.com/@eddmpython', label: 'YouTube' },
		{ icon: Mail, href: 'mailto:eddmpython@gmail.com', label: 'Email' }
	];
</script>

<footer class="bg-vx-bg-darker border-t border-vx-border pt-14 pb-8 px-6">
	<div class="max-w-6xl mx-auto">
		<div class="grid grid-cols-1 md:grid-cols-5 gap-8 mb-10">
			<div class="md:col-span-1">
				<div class="flex items-center gap-1.5 mb-3">
					<img src="{base}/icon-final.png" alt="Vectrix" width="28" height="28" />
					<strong class="text-vx-text text-lg">Vectrix</strong>
				</div>
				<p class="text-sm text-vx-text-dim leading-relaxed mb-4">Zero-config time series forecasting for Python.</p>
				<div class="flex gap-2 mb-4">
					{#each socialLinks as link}
						<a href={link.href} target="_blank" rel="noopener" title={link.label}
							class="w-9 h-9 rounded-lg bg-vx-bg-card border border-vx-border flex items-center justify-center text-vx-text-dim hover:text-vx-primary hover:border-vx-primary transition-all">
							<link.icon class="w-4 h-4" />
						</a>
					{/each}
				</div>
				<a href="https://buymeacoffee.com/eddmpython" target="_blank" rel="noopener">
					<img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="160" height="40" class="hover:opacity-90 transition-opacity" />
				</a>
			</div>

			{#each Object.entries(footerLinks) as [title, links]}
				<div>
					<h5 class="text-xs font-semibold uppercase tracking-wider text-vx-text-muted mb-4">{title}</h5>
					<div class="flex flex-col gap-2">
						{#each links as link}
							<a href={link.href}
								target={link.href.startsWith('http') ? '_blank' : undefined}
								rel={link.href.startsWith('http') ? 'noopener' : undefined}
								class="text-sm text-vx-text-dim hover:text-vx-primary transition-colors no-underline">
								{link.label}
							</a>
						{/each}
					</div>
				</div>
			{/each}
		</div>

		<div class="flex flex-col md:flex-row justify-between items-center pt-6 border-t border-vx-border text-xs text-vx-text-dim gap-2">
			<span>MIT License &copy; 2026 eddmpython</span>
			<span>Made with NumPy + SciPy + Pandas</span>
		</div>
	</div>
</footer>
