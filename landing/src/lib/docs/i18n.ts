import { writable } from 'svelte/store';

export type Locale = 'en' | 'ko';

export const locale = writable<Locale>('en');

export function getLocaleFromPath(path: string): Locale {
	if (path.includes('/ko/')) return 'ko';
	return 'en';
}
