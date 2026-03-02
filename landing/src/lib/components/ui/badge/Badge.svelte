<script lang="ts">
	import { cn } from '$lib/utils';
	import type { Snippet } from 'svelte';

	type Variant = 'default' | 'accent' | 'success' | 'warning';

	interface Props {
		variant?: Variant;
		class?: string;
		children: Snippet;
	}

	let { variant = 'default', class: className, children }: Props = $props();

	const variants: Record<Variant, string> = {
		default: 'bg-vx-primary/10 text-vx-primary-light border-vx-primary/20',
		accent: 'bg-vx-accent/10 text-vx-accent-light border-vx-accent/20',
		success: 'bg-vx-success/10 text-vx-success border-vx-success/20',
		warning: 'bg-vx-warning/10 text-vx-warning border-vx-warning/20'
	};
</script>

<span class={cn(
	'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border',
	variants[variant],
	className
)}>
	{@render children()}
</span>
