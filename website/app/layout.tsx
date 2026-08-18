import type { Metadata, Viewport } from "next";
import "./globals.css";

const title =
  "GRNEdit — Efficient General Video Editing with Binary Evidence";
const description =
  "GRNEdit is an efficient general video editing framework from a binary-evidence perspective, competitive with leading editing models with under 3% additional parameters.";
const publicBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const metadata: Metadata = {
  metadataBase: new URL("https://foxerity.github.io/GRNEdit/"),
  title,
  description,
  applicationName: "GRNEdit",
  authors: [{ name: "Feng Xie" }],
  keywords: [
    "GRNEdit",
    "video editing",
    "generative refinement networks",
    "binary evidence",
    "video generation",
  ],
  alternates: {
    canonical: "https://foxerity.github.io/GRNEdit/",
  },
  openGraph: {
    type: "website",
    url: "https://foxerity.github.io/GRNEdit/",
    siteName: "GRNEdit",
    title,
    description,
    images: [
      {
        url: "https://foxerity.github.io/GRNEdit/og.png",
        width: 1200,
        height: 630,
        alt: "GRNEdit — efficient general video editing with binary evidence",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: ["https://foxerity.github.io/GRNEdit/og.png"],
  },
  icons: {
    icon: `${publicBasePath}/favicon.svg`,
    shortcut: `${publicBasePath}/favicon.svg`,
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#08110f",
  colorScheme: "dark",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
