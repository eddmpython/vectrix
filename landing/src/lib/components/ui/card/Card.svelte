<script lang="ts">
	import { cn } from '$lib/utils';
	import type { Snippet } from 'svelte';

	interface Props {
		class?: string;
		hover?: boolean;
		children: Snippet;
	}

	let { class: className, hover = true, children }: Props = $props();
</script>

<div class={cn(
	'rounded-xl bg-vx-bg-card border border-vx-border p-6 transition-all duration-300',
	hover && 'hover:border-vx-primary/50 hover:-translate-y-1 hover:shadow-lg hover:shadow-vx-primary/5',
	className
)}>
	{@render children()}
</div>
