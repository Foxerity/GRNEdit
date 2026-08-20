# GRNEdit project website

The source for the GRNEdit project page lives in this directory. It uses
Vinext/React for the page and native MP4 playback for the result gallery.

## Local development

```bash
cd docs
npm ci
npm run dev
```

Open `http://localhost:3000`.

## Production build

Build the site locally before committing a change:

```bash
npm run build
```

To reproduce the complete GitHub Pages artifact locally, run:

```bash
npm run export:pages
```

The command builds the static site into `dist/client` and verifies every
rendered URL, video, poster, figure, logo, and refinement asset.

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

The workflow in [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)
builds and deploys the page after each push to `main`. Generated HTML, CSS, and
JavaScript are uploaded as a temporary Pages artifact and are not committed to
the repository.

Before pushing a page update, verify the same artifact locally with:

```bash
npm run export:pages
```

In **Settings → Pages**, the publishing source must be set to **GitHub Actions**.
No secret or additional deployment branch is required. The project URL is:

`https://foxerity.github.io/GRNEdit/`
