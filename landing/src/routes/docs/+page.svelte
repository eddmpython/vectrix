<script lang="ts">
	import { base } from '$app/paths';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	onMount(() => {
		goto(`${base}/docs/getting-started/installation`);
	});
</script>

<div style="padding: 2rem; color: #94a3b8;">
	<p>Redirecting to documentation...</p>
</div>
