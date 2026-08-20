import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const html = await readFile(
    new URL("../dist/client/index.html", import.meta.url),
    "utf8",
  );

  return new Response(html, {
    headers: { "content-type": "text/html; charset=utf-8" },
  });
}

test("renders the GRNEdit project page with video results", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>GRNEdit/);
  assert.match(
    html,
    /GRNEdit: Efficient General Video Editing from a New Binary-Evidence/,
  );
  assert.equal(
    (html.match(/class="title-line"/g) ?? []).length,
    2,
    "the desktop paper title should be split into exactly two lines",
  );
  assert.doesNotMatch(html, /Editing starts/);
  assert.match(html, /under 3% extra parameters/i);
  assert.match(html, /Prediction and evidence, step by step/);
  assert.match(html, /Cite GRNEdit/);
  assert.match(html, /arXiv:2608\.16328/);
  assert.match(html, /https:\/\/arxiv\.org\/pdf\/2608\.16328/);
  assert.doesNotMatch(html, /2608\.00000/);
  assert.match(html, /@article\{xie2026grnedit/);
  assert.match(html, /Open-source plan/);
  assert.match(html, /class="release-timeline"/);
  assert.match(html, /class="release-marker"[^>]*>01</);
  assert.doesNotMatch(html, /class="release-item/);
  assert.match(html, /data-reveal="true"/);
  assert.doesNotMatch(html, /04 · Open-source release/);
  assert.doesNotMatch(html, /href="#release"/);
  assert.doesNotMatch(html, /id="release"/);
  assert.match(html, /global-style-watercolor\.mp4/);
  assert.match(html, /creative-alchemical-symbols\.mp4/);
  assert.match(html, /step-010\.mp4/);
  assert.match(html, /margin-010\.mp4/);
  assert.match(html, /step-010\.webp/);
  assert.match(html, /preload="auto"/);
  assert.match(html, /Step 1 → Step 50/);
  for (const step of [1, 10, 20, 30, 40, 50]) {
    assert.match(html, new RegExp(`<span>Step</span><strong>${step}</strong>`));
  }
  assert.doesNotMatch(html, /Step 0 → Step 49/);
  assert.doesNotMatch(html, /aria-label="Refinement step"/);
  assert.ok(
    html.indexOf("global-style-watercolor.mp4") <
      html.indexOf("creative-alchemical-symbols.mp4"),
    "the global-style case should be the featured hero video",
  );
});

test("ships fast-start refinement videos and every first-frame poster", async () => {
  const cases = [
    "0000-global-style",
    "0070-local-change",
    "0127-background",
    "0182-removal",
    "0287-addition",
    "0323-creative",
    "0385-subtitle",
  ];
  const steps = ["000", "010", "020", "029", "039", "049"];

  const mediaPairs = cases.flatMap((caseName) =>
    steps.flatMap((step) =>
      ["step", "margin"].map((kind) => ({ caseName, kind, step })),
    ),
  );

  await Promise.all(
    mediaPairs.map(async ({ caseName, kind, step }) => {
      const mediaRoot = `../public/media/refinement/${caseName}/${kind}-${step}`;
      const video = await readFile(new URL(`${mediaRoot}.mp4`, import.meta.url));
      const movieIndex = video.indexOf("moov");
      const mediaIndex = video.indexOf("mdat");

      assert.ok(movieIndex > 0, `${caseName}/${kind}-${step} has a movie index`);
      assert.ok(mediaIndex > 0, `${caseName}/${kind}-${step} has media data`);
      assert.ok(
        movieIndex < mediaIndex,
        `${caseName}/${kind}-${step} places its movie index before media data`,
      );
      await access(new URL(`${mediaRoot}.webp`, import.meta.url));
    }),
  );
});

test("ships all four MP4 result assets", async () => {
  const videos = [
    "global-style-watercolor.mp4",
    "background-library-fireplace.mp4",
    "remove-person-behind-computer.mp4",
    "creative-alchemical-symbols.mp4",
  ];

  await Promise.all(
    videos.map((name) =>
      access(new URL(`../public/media/videos/${name}`, import.meta.url)),
    ),
  );
});
