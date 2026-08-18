import assert from "node:assert/strict";
import { access } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
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
  assert.doesNotMatch(html, /aria-label="Refinement step"/);
  assert.ok(
    html.indexOf("global-style-watercolor.mp4") <
      html.indexOf("creative-alchemical-symbols.mp4"),
    "the global-style case should be the featured hero video",
  );
});

test("ships every matched refinement and delta-margin pair", async () => {
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

  await Promise.all(
    cases.flatMap((caseName) =>
      steps.flatMap((step) =>
        ["step", "margin"].map((kind) =>
          access(
            new URL(
              `../public/media/refinement/${caseName}/${kind}-${step}.mp4`,
              import.meta.url,
            ),
          ),
        ),
      ),
    ),
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
