<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { ShoppingBag, DollarSign, Zap, Activity, Users, Monitor } from 'lucide-svelte';

	const cases = [
		{ icon: ShoppingBag, color: 'text-vx-primary bg-vx-primary/10', title: 'Retail & E-Commerce', desc: 'Demand forecasting, inventory optimization, seasonal planning with intermittent demand support (Croston, ADIDA).' },
		{ icon: DollarSign, color: 'text-vx-accent bg-vx-accent/10', title: 'Finance & Trading', desc: 'Volatility modeling (GARCH), regime detection for market states, probabilistic risk assessment with confidence intervals.' },
		{ icon: Zap, color: 'text-vx-success bg-vx-success/10', title: 'Energy & Utilities', desc: 'Load forecasting with multi-seasonal patterns (hourly/daily/weekly). MSTL and TBATS handle complex seasonality.' },
		{ icon: Activity, color: 'text-vx-warning bg-vx-warning/10', title: 'Healthcare & IoT', desc: 'Patient volume prediction, sensor data forecasting, anomaly detection with self-healing for drift correction.' },
		{ icon: Users, color: 'text-vx-primary bg-vx-primary/10', title: 'HR & Workforce', desc: 'Staffing demand prediction, attrition forecasting. Hierarchical reconciliation for department-level planning.' },
		{ icon: Monitor, color: 'text-vx-accent bg-vx-accent/10', title: 'SaaS & Product', desc: 'User growth modeling, revenue forecasting, churn prediction. Business constraints enforce realistic growth bounds.' }
	];
</script>

<section class="bg-vx-bg-darker py-20 px-6">
	<div class="max-w-6xl mx-auto text-center">
		<span class="text-xs font-semibold uppercase tracking-widest text-vx-primary mb-3 block">Use Cases</span>
		<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-vx-text mb-3">Built for real-world applications</h2>
		<p class="text-vx-text-muted max-w-xl mx-auto mb-12">
			From demand planning to financial modeling — Vectrix adapts to your domain.
		</p>

		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
			{#each cases as c}
				<Card class="text-left">
					<div class="w-12 h-12 rounded-xl flex items-center justify-center mb-4 {c.color}">
						<c.icon class="w-6 h-6" />
					</div>
					<h4 class="text-base font-semibold text-vx-text mb-2">{c.title}</h4>
					<p class="text-sm text-vx-text-muted leading-relaxed">{c.desc}</p>
				</Card>
			{/each}
		</div>
	</div>
</section>
