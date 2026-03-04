import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type Plugin } from 'vite';
import path from 'path';
import fs from 'fs';

function blogAssetsPlugin(): Plugin {
	const src = path.resolve(__dirname, '..', 'docs', 'blog', 'assets');
	const dest = path.resolve(__dirname, 'static', 'blog');

	function syncAssets() {
		if (!fs.existsSync(src)) return;
		if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
		for (const file of fs.readdirSync(src)) {
			fs.copyFileSync(path.join(src, file), path.join(dest, file));
		}
	}

	return {
		name: 'blog-assets-sync',
		buildStart() { syncAssets(); },
		configureServer(server) {
			syncAssets();
			server.watcher.add(src);
			server.watcher.on('change', (p) => {
				if (p.startsWith(src)) syncAssets();
			});
			server.watcher.on('add', (p) => {
				if (p.startsWith(src)) syncAssets();
			});
		}
	};
}

export default defineConfig({
	plugins: [blogAssetsPlugin(), tailwindcss(), sveltekit()],
	resolve: {
		alias: {
			'@docs': path.resolve(__dirname, '..', 'docs')
		}
	},
	server: {
		fs: {
			allow: [path.resolve(__dirname, '..', 'docs')]
		}
	}
});
