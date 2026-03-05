# Blog Assets — SVG Serving Guide

## Architecture

Blog posts are rendered by **SvelteKit** (`/blog/[slug]` route), not MkDocs.
SVG images must be served as **static files** through SvelteKit.

## Why Not MkDocs?

SvelteKit has a catch-all route at `/docs/[...slug]/+page.ts`.
When SVGs are placed only in `docs/blog/assets/` and served via MkDocs (`/vectrix/docs/blog/assets/*.svg`),
the catch-all route intercepts the request and prerenders it as `.svg.html` → **404**.

```
Request: /vectrix/docs/blog/assets/forecasting-hero.svg
         ↓
SvelteKit /docs/[...slug] catches it
         ↓
Prerenders as forecasting-hero.svg.html
         ↓
Original .svg never served → 404
```

## Correct Setup

SVGs live in **two places** (source of truth is here, copy is in SvelteKit static):

```
docs/blog/assets/*.svg          ← source files (this folder)
landing/static/blog/assets/*.svg ← copy for SvelteKit static serving
```

Blog md files reference: `/vectrix/blog/assets/<name>.svg`

SvelteKit `static/` files are served directly without routing, so `/vectrix/blog/assets/*.svg` resolves to `landing/static/blog/assets/*.svg` at build time.

## Adding a New SVG

1. Create the SVG in `docs/blog/assets/`
2. Copy it to `landing/static/blog/assets/`
3. Reference in md as `![alt text](/vectrix/blog/assets/<name>.svg)`

## Path Reference

| Wrong | Right |
|---|---|
| `/vectrix/docs/blog/assets/foo.svg` | `/vectrix/blog/assets/foo.svg` |
| `/vectrix/blog/foo.svg` | `/vectrix/blog/assets/foo.svg` |
| `./assets/foo.svg` (relative) | `/vectrix/blog/assets/foo.svg` |
