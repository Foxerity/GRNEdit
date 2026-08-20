import type { NextConfig } from "next";

const staticExport = process.env.GRNEDIT_STATIC_EXPORT === "1";
const basePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? "").replace(/\/$/, "");
const pagesOrigin = process.env.GRNEDIT_PAGES_ORIGIN ?? "https://foxerity.github.io";

const nextConfig: NextConfig = {
  ...(staticExport ? { output: "export" as const } : {}),
  // GitHub Pages mounts the exported artifact at /GRNEdit. Keeping the route
  // itself at / avoids a duplicated /GRNEdit/GRNEdit path. An absolute asset
  // prefix keeps the emitted _next directory at the artifact root while its
  // URLs still point to the GitHub project path.
  basePath: staticExport ? "" : basePath,
  assetPrefix: staticExport && basePath ? `${pagesOrigin}${basePath}` : basePath,
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
