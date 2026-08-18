# GRNEdit project website

The source for the GRNEdit project page lives in this directory. It uses
Vinext/React for the page and native MP4 playback for the result gallery.

## Local development

```bash
cd website
npm ci
npm run dev
```

Open `http://localhost:3000`.

## Production build

Build the site locally before committing a change:

```bash
npm run build
```

To reproduce the GitHub Pages artifact locally, set the repository base path
and enable static export before building:

```bash
GRNEDIT_STATIC_EXPORT=1 NEXT_PUBLIC_BASE_PATH=/GRNEdit npm run build
```

The exported static site is written to `dist/client`.

## Editing content and media

- Main page copy, authors, result cards, links, and citation:
  [`app/page.tsx`](app/page.tsx)
- Browser title and social metadata: [`app/layout.tsx`](app/layout.tsx)
- Colors, spacing, typography, and responsive layout:
  [`app/globals.css`](app/globals.css)
- Refinement case names and step/margin presentation:
  [`app/components/RefinementExplorer.tsx`](app/components/RefinementExplorer.tsx)
- Videos, posters, figures, and affiliation logos: [`public/media`](public/media)

The simplest way to replace an asset is to keep its existing filename and file
type. If the filename changes, update the matching path in `app/page.tsx` or
`app/components/RefinementExplorer.tsx`. After editing, keep `npm run dev`
running to preview changes, then run `npm run export:pages` to refresh the
committed GitHub Pages files.

## GitHub Pages

The complete static site is committed under [`../docs`](../docs), including
HTML, CSS, JavaScript, figures, logos, and MP4 files. Regenerate it after a
website change with:

```bash
npm run export:pages
```

Once the repository can be public, open **Settings → Pages**, choose
**Deploy from a branch**, then select **main** and **/docs**. No build service,
secret, or additional deployment branch is required. The project URL is then:

`https://foxerity.github.io/GRNEdit/`
