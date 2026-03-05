import { readFileSync } from 'fs';
import { resolve } from 'path';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const jsonPath = resolve('static/playground/data.json');
	const raw = readFileSync(jsonPath, 'utf-8');
	const playgroundData = JSON.parse(raw);
	return { playgroundData };
};
