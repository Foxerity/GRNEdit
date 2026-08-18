import { spawnSync } from "node:child_process";
import {
  cp,
  readdir,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const websiteDir = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const repositoryDir = path.resolve(websiteDir, "..");
const distDir = path.resolve(websiteDir, "dist", "client");
const docsDir = path.resolve(repositoryDir, "docs");
const publicDir = path.resolve(websiteDir, "public");

if (
  path.dirname(docsDir) !== repositoryDir ||
  path.basename(docsDir) !== "docs"
) {
  throw new Error(`Refusing to replace unexpected Pages directory: ${docsDir}`);
}

await rm(path.resolve(websiteDir, "dist"), { recursive: true, force: true });

const buildEnvironment = {
  ...process.env,
  GRNEDIT_STATIC_EXPORT: "1",
  NEXT_PUBLIC_BASE_PATH: "/GRNEdit",
  GRNEDIT_PAGES_ORIGIN: "https://foxerity.github.io",
};

const command = process.platform === "win32"
  ? process.env.ComSpec ?? "cmd.exe"
  : "npm";
const args = process.platform === "win32"
  ? ["/d", "/s", "/c", "npm run build"]
  : ["run", "build"];

const build = spawnSync(command, args, {
  cwd: websiteDir,
  env: buildEnvironment,
  stdio: "inherit",
});

if (build.error) throw build.error;

const expectedWindowsTeardownCodes = new Set([-1073740791, 3221226505]);
const indexPath = path.join(distDir, "index.html");
let indexHtml = "";

try {
  indexHtml = await readFile(indexPath, "utf8");
} catch {
  // The build result below provides the actionable failure message.
}

const toleratedWindowsTeardown =
  process.platform === "win32" &&
  expectedWindowsTeardownCodes.has(build.status) &&
  indexHtml.length > 0;

if (build.status !== 0 && !toleratedWindowsTeardown) {
  throw new Error(`GitHub Pages export failed with exit code ${build.status}.`);
}

const requiredFiles = [
  "index.html",
  "index.rsc",
  "favicon.svg",
  path.join("media", "videos", "global-style-watercolor.mp4"),
  path.join(
    "media",
    "refinement",
    "0000-global-style",
    "margin-049.mp4",
  ),
];

for (const relativePath of requiredFiles) {
  await stat(path.join(distDir, relativePath));
}

if (
  !indexHtml.includes("/GRNEdit/media/") ||
  !indexHtml.includes("https://foxerity.github.io/GRNEdit/")
) {
  throw new Error("Static HTML is missing the expected GitHub Pages base path.");
}

await rm(docsDir, { recursive: true, force: true });
await cp(distDir, docsDir, { recursive: true, force: true });
await writeFile(path.join(docsDir, ".nojekyll"), "", "utf8");

await stat(path.join(docsDir, "_next"));

async function createManifest(root, directory = root) {
  const manifest = new Map();

  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      const nested = await createManifest(root, entryPath);
      for (const [relativePath, size] of nested) {
        manifest.set(relativePath, size);
      }
    } else if (entry.isFile()) {
      const relativePath = path.relative(root, entryPath).replaceAll("\\", "/");
      manifest.set(relativePath, (await stat(entryPath)).size);
    }
  }

  return manifest;
}

// Every source asset must be present byte-for-byte in the committed Pages tree.
// This guards the large video collection as well as logos, posters, and images.
const publicManifest = await createManifest(publicDir);
const docsManifest = await createManifest(docsDir);
const assetProblems = [];

for (const [relativePath, size] of publicManifest) {
  const exportedSize = docsManifest.get(relativePath);
  if (exportedSize === undefined) {
    assetProblems.push(`missing: ${relativePath}`);
  } else if (exportedSize !== size) {
    assetProblems.push(
      `size mismatch: ${relativePath} (${size} -> ${exportedSize})`,
    );
  }
}

if (assetProblems.length > 0) {
  throw new Error(
    `Static export did not preserve every public asset:\n${assetProblems.slice(0, 20).join("\n")}`,
  );
}

// Check the concrete script, stylesheet, image, and video URLs emitted by the
// rendered homepage. GitHub Pages serves these below /GRNEdit/.
const emittedReferences = [
  ...indexHtml.matchAll(/(?:href|src)="(\/GRNEdit\/[^"?#]+)(?:[?#][^"]*)?"/g),
].map((match) => decodeURI(match[1].slice("/GRNEdit/".length)));

const missingReferences = [...new Set(emittedReferences)].filter(
  (relativePath) => !docsManifest.has(relativePath),
);

if (missingReferences.length > 0) {
  throw new Error(
    `Rendered homepage references missing files:\n${missingReferences.slice(0, 20).join("\n")}`,
  );
}

async function summarize(directory) {
  let files = 0;
  let bytes = 0;

  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      const nested = await summarize(entryPath);
      files += nested.files;
      bytes += nested.bytes;
    } else if (entry.isFile()) {
      files += 1;
      bytes += (await stat(entryPath)).size;
    }
  }

  return { files, bytes };
}

const summary = await summarize(docsDir);
console.log(
  `GitHub Pages export ready in docs/ (${summary.files} files, ${(summary.bytes / 1024 / 1024).toFixed(2)} MiB; ${publicManifest.size} public assets verified).`,
);
