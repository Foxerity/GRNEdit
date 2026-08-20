import { ResultVideo } from "./components/ResultVideo";
import { RefinementExplorer } from "./components/RefinementExplorer";
import { ScrollReveal } from "./components/ScrollReveal";
import Image from "next/image";

export const dynamic = "force-static";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const asset = (path: string) => `${basePath}${path}`;

const results = [
  {
    index: "01",
    category: "Global style",
    title: "Watercolor animation",
    prompt: "Apply the watercolor animation style.",
    description:
      "A global appearance edit that keeps the source composition and motion legible.",
    video: "/media/videos/global-style-watercolor.mp4",
    poster: "/media/posters/global-style-watercolor.jpg",
  },
  {
    index: "02",
    category: "Background",
    title: "Library & fireplace",
    prompt: "Replace the background with a classic library and fireplace.",
    description:
      "A scene-level replacement that preserves foreground identity and temporal content.",
    video: "/media/videos/background-library-fireplace.mp4",
    poster: "/media/posters/background-library-fireplace.jpg",
  },
  {
    index: "03",
    category: "Local removal",
    title: "Remove a person",
    prompt: "Remove the man behind the computer.",
    description:
      "A localized removal that leaves the primary subject and surrounding scene intact.",
    video: "/media/videos/remove-person-behind-computer.mp4",
    poster: "/media/posters/remove-person-behind-computer.jpg",
  },
  {
    index: "04",
    category: "Creative edit",
    title: "Alchemical symbols",
    prompt: "Remove the flame and add glowing alchemical symbols.",
    description:
      "A compositional edit combining removal and addition in one instruction.",
    video: "/media/videos/creative-alchemical-symbols.mp4",
    poster: "/media/posters/creative-alchemical-symbols.jpg",
  },
];

const authors = [
  { name: "Feng Xie", marks: "1,2,*" },
  { name: "Jiagao Hu", marks: "2" },
  { name: "Fuhao Li", marks: "2" },
  { name: "Zepeng Wang", marks: "2" },
  { name: "Yuxuan Chen", marks: "2" },
  { name: "Dahua Gao", marks: "1,†" },
  { name: "Fei Wang", marks: "2" },
  { name: "Daiguo Zhou", marks: "2" },
];

const releaseMilestones = [
  { index: "01", label: "Inference code", status: "Available", complete: true },
  { index: "02", label: "Preprint", status: "Available", complete: true },
  { index: "03", label: "Training code", status: "Planned", complete: false },
  { index: "04", label: "Model weights", status: "Planned", complete: false },
];

const arxivId = "2608.16328";
const arxivPaperUrl = `https://arxiv.org/pdf/${arxivId}`;
const bibtexCitation = `@article{xie2026grnedit,
  title         = {GRNEdit: Efficient General Video Editing from a New Binary-Evidence Perspective in Generative Refinement Networks},
  author        = {Xie, Feng and Hu, Jiagao and Li, Fuhao and Wang, Zepeng and Chen, Yuxuan and Gao, Dahua and Wang, Fei and Zhou, Daiguo},
  journal       = {arXiv preprint arXiv:${arxivId}},
  year          = {2026},
  eprint        = {${arxivId}},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}`;

function ArrowIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d="M5 15 15 5M7 5h8v8" />
    </svg>
  );
}

function GithubIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.6a9.6 9.6 0 0 0-3.04 18.71c.48.09.66-.2.66-.46v-1.69c-2.68.58-3.25-1.14-3.25-1.14-.44-1.12-1.08-1.42-1.08-1.42-.88-.6.07-.59.07-.59.97.07 1.49 1 1.49 1 .86 1.48 2.26 1.05 2.81.8.09-.63.34-1.05.61-1.3-2.14-.24-4.39-1.07-4.39-4.75 0-1.05.38-1.91 1-2.58-.1-.24-.43-1.22.09-2.55 0 0 .81-.26 2.64.99A9.2 9.2 0 0 1 12 7.3a9.2 9.2 0 0 1 2.4.32c1.83-1.25 2.64-.99 2.64-.99.52 1.33.19 2.31.09 2.55.62.67 1 1.53 1 2.58 0 3.69-2.26 4.5-4.4 4.74.35.3.65.88.65 1.78v2.57c0 .26.18.55.66.46A9.6 9.6 0 0 0 12 2.6Z" />
    </svg>
  );
}

export default function Home() {
  return (
    <>
      <ScrollReveal />
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>

      <header className="site-header">
        <nav className="nav shell" aria-label="Primary navigation">
          <a className="wordmark" href="#top" aria-label="GRNEdit home">
            <span className="wordmark-mark" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            GRN<span>Edit</span>
          </a>
          <div className="nav-links">
            <a href="#results">Results</a>
            <a href="#method">Method</a>
            <a href="#refinement">Refinement</a>
            <a href="#citation">Citation</a>
          </div>
          <a
            className="nav-github"
            href="https://github.com/Foxerity/GRNEdit"
            target="_blank"
            rel="noreferrer"
          >
            <GithubIcon />
            <span>GitHub</span>
          </a>
        </nav>
      </header>

      <main id="main-content">
        <section className="paper-hero shell" id="top">
          <div className="paper-identity">
            <h1>
              <span className="title-line">
                GRNEdit: Efficient General Video Editing from a New
              </span>
              <span className="title-line">
                Binary-Evidence Perspective in Generative Refinement Networks
              </span>
            </h1>

            <div className="hero-authors" aria-label="Authors">
              {authors.map((author) => (
                <span key={author.name}>
                  {author.name}<sup>{author.marks}</sup>
                </span>
              ))}
            </div>

            <div className="hero-affiliations" aria-label="Affiliations">
              <div>
                <Image
                  src={asset("/media/institutions/xidian-university.png")}
                  alt="Xidian University"
                  width={230}
                  height={62}
                  priority
                />
                <span><sup>1</sup> Xidian University</span>
              </div>
              <div>
                <Image
                  src={asset("/media/institutions/xiaomi.svg")}
                  alt="Xiaomi"
                  width={512}
                  height={512}
                  priority
                />
                <span><sup>2</sup> MiLM Plus, Xiaomi Inc.</span>
              </div>
            </div>

            <p className="hero-note">
              <sup>*</sup> This work was completed during Feng Xie&apos;s internship
              at Xiaomi. We thank Xiaomi for its support. <sup>†</sup> Corresponding
              author.
            </p>
          </div>

          <div className="hero-showcase">
            <ResultVideo
              src={asset("/media/videos/global-style-watercolor.mp4")}
              poster={asset("/media/posters/global-style-watercolor.jpg")}
              label="Global video edit: apply a watercolor animation style"
              featured
            />
          </div>

          <div className="hero-summary">
            <p>
              GRNEdit is an efficient general video editing framework from a
              binary-evidence perspective, offering performance competitive with
              leading video editing models while adding under 3% extra parameters.
            </p>
            <div className="hero-actions">
              <a
                className="button button-primary"
                href="https://github.com/Foxerity/GRNEdit"
                target="_blank"
                rel="noreferrer"
              >
                <GithubIcon /> Explore the code
              </a>
              <a
                className="button button-ghost"
                href={arxivPaperUrl}
                target="_blank"
                rel="noreferrer"
              >
                Read the paper <ArrowIcon />
              </a>
            </div>
          </div>
        </section>

        <section className="release-strip" aria-label="Open-source release status">
          <div className="shell release-strip-inner" data-reveal>
            <span className="release-label">Open-source plan</span>
            <ol className="release-timeline">
              {releaseMilestones.map((milestone) => (
                <li
                  className={milestone.complete ? "is-complete" : undefined}
                  key={milestone.index}
                >
                  <span className="release-marker" aria-hidden="true">
                    {milestone.index}
                  </span>
                  <strong>{milestone.label}</strong>
                  <small>{milestone.status}</small>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section className="section shell" id="results">
          <div className="section-heading" data-reveal>
            <div>
              <p className="section-index">01 · Video results</p>
              <h2>One framework.<br />Many kinds of edits.</h2>
            </div>
          </div>

          <div className="results-grid">
            {results.map((result) => (
              <article className="result-card" data-reveal key={result.index}>
                <div className="result-video-wrap">
                  <ResultVideo
                    src={asset(result.video)}
                    poster={asset(result.poster)}
                    label={`${result.category}: ${result.prompt}`}
                  />
                  <span className="result-index">{result.index}</span>
                </div>
                <div className="result-copy">
                  <p className="result-category">{result.category}</p>
                  <h3>{result.title}</h3>
                  <p className="result-prompt">“{result.prompt}”</p>
                  <p className="result-description">{result.description}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="section method-section" id="method">
          <div className="shell">
            <div className="section-heading method-heading" data-reveal>
              <div>
                <p className="section-index">02 · Binary-evidence perspective</p>
                <h2>Resolve the edit,<br />progressively.</h2>
              </div>
            </div>

            <div className="method-steps">
              <article data-reveal>
                <span>01</span>
                <div className="method-glyph bit-glyph" aria-hidden="true">
                  <i>1</i><i>0</i><i>1</i><i>0</i>
                </div>
                <h3>Locate with evidence</h3>
                <p>
                  Early binary evidence separates editable scope from content
                  that should remain stable.
                </p>
              </article>
              <article data-reveal>
                <span>02</span>
                <div className="method-glyph refine-glyph" aria-hidden="true">
                  <i /><i /><i /><i />
                </div>
                <h3>Refine generatively</h3>
                <p>
                  Coarse proposals evolve into target semantics, coherent
                  structure, and fine details.
                </p>
              </article>
              <article data-reveal>
                <span>03</span>
                <div className="method-glyph efficiency-glyph" aria-hidden="true">
                  <strong>&lt;3%</strong>
                </div>
                <h3>Adapt efficiently</h3>
                <p>
                  The editing capability is introduced with under 3% additional
                  parameters over the GRN backbone.
                </p>
              </article>
            </div>

            <div className="refinement-block" id="refinement">
              <div className="refinement-heading" data-reveal>
                <div>
                  <p className="figure-kicker">Inside the refinement process</p>
                  <h3>Prediction and evidence, step by step.</h3>
                </div>
              </div>
              <RefinementExplorer basePath={basePath} />
            </div>
          </div>
        </section>

        <section className="section shell qualitative-section">
          <div className="section-heading compact-heading" data-reveal>
            <div>
              <p className="section-index">03 · Qualitative gallery</p>
              <h2>Broad coverage,<br />consistent intent.</h2>
            </div>
          </div>

          <figure className="gallery-figure gallery-figure-wide">
            <a
              href={asset("/media/figures/qualitative-results-across-tasks.webp")}
              target="_blank"
              rel="noreferrer"
              aria-label="Open qualitative results across tasks at full size"
            >
              <Image
                src={asset("/media/figures/qualitative-results-across-tasks.webp")}
                alt="Qualitative GRNEdit results for style, background, and local edits"
                width={2560}
                height={1286}
                loading="lazy"
              />
            </a>
            <figcaption>
              <span>Across editing tasks</span>
              <p>Style · Background · Local change</p>
            </figcaption>
          </figure>

          <figure className="gallery-figure gallery-figure-wide">
            <a
              href={asset("/media/figures/qualitative-results-generalization.webp")}
              target="_blank"
              rel="noreferrer"
              aria-label="Open qualitative generalization results at full size"
            >
              <Image
                src={asset("/media/figures/qualitative-results-generalization.webp")}
                alt="Qualitative GRNEdit results for removal, subtitle removal, and addition"
                width={2560}
                height={1286}
                loading="lazy"
              />
            </a>
            <figcaption>
              <span>Generalization & composition</span>
              <p>Removal · Subtitle removal · Addition</p>
            </figcaption>
          </figure>
        </section>

        <section className="citation-section" id="citation">
          <div className="shell citation-inner">
            <div className="citation-copy">
              <p className="section-index">05 · Citation</p>
              <h2>Cite GRNEdit</h2>
              <p>
                If you find GRNEdit useful, please cite our{" "}
                <a href={arxivPaperUrl} target="_blank" rel="noreferrer">
                  arXiv paper
                </a>.
              </p>
            </div>
            <div className="citation-code" data-reveal>
              <div>
                <span>BibTeX</span>
                <small>arXiv · {arxivId}</small>
              </div>
              <pre><code>{bibtexCitation}</code></pre>
            </div>
          </div>
        </section>

        <section className="acknowledgement">
          <div className="shell acknowledgement-inner">
            <p className="section-index">Acknowledgements</p>
            <p>
              GRNEdit is built upon <a href="https://arxiv.org/abs/2604.13030" target="_blank" rel="noreferrer">GRN: Generative Refinement Networks for Visual Synthesis</a> and its <a href="https://github.com/bytedance/GRN" target="_blank" rel="noreferrer">official codebase</a>. We sincerely thank the authors for releasing their code and for their exciting work.
            </p>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="shell footer-inner">
          <a className="wordmark footer-wordmark" href="#top">
            GRN<span>Edit</span>
          </a>
          <p>Efficient general video editing from a binary-evidence perspective.</p>
          <div>
            <a href={arxivPaperUrl} target="_blank" rel="noreferrer">arXiv</a>
            <a href="https://github.com/Foxerity/GRNEdit" target="_blank" rel="noreferrer">GitHub</a>
            <a href="https://github.com/Foxerity/GRNEdit/blob/main/LICENSE" target="_blank" rel="noreferrer">MIT License</a>
          </div>
        </div>
      </footer>
    </>
  );
}
