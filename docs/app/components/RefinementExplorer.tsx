"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type RefinementExplorerProps = {
  basePath: string;
};

const cases = [
  {
    id: "0000-global-style",
    label: "Global style",
    prompt: "Transform the video into an Impressionist painting style.",
  },
  {
    id: "0070-local-change",
    label: "Local change",
    prompt: "Change the green T-shirt into a navy business suit.",
  },
  {
    id: "0127-background",
    label: "Background",
    prompt: "Replace the background with a French bistro.",
  },
  {
    id: "0182-removal",
    label: "Removal",
    prompt: "Remove the blonde woman from the scene.",
  },
  {
    id: "0287-addition",
    label: "Addition",
    prompt: "Add a street lamp beside the subject.",
  },
  {
    id: "0323-creative",
    label: "Creative edit",
    prompt: "Turn the corridor walls into infinite bookshelves and lights.",
  },
  {
    id: "0385-subtitle",
    label: "Subtitle removal",
    prompt: "Remove the subtitles at the bottom of the video.",
  },
] as const;

const sampledSteps = [
  { assetStep: "000", displayStep: 1 },
  { assetStep: "010", displayStep: 10 },
  { assetStep: "020", displayStep: 20 },
  { assetStep: "029", displayStep: 30 },
  { assetStep: "039", displayStep: 40 },
  { assetStep: "049", displayStep: 50 },
] as const;
const mediaKinds = ["step", "margin"] as const;
const videoCount = sampledSteps.length * 2;

const refinementMediaPath = (
  basePath: string,
  caseId: (typeof cases)[number]["id"],
  kind: (typeof mediaKinds)[number],
  step: (typeof sampledSteps)[number]["assetStep"],
  extension: "mp4" | "webp",
) =>
  `${basePath}/media/refinement/${caseId}/${kind}-${step}.${extension}`;

export function RefinementExplorer({ basePath }: RefinementExplorerProps) {
  const [caseIndex, setCaseIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [readyVideoIds, setReadyVideoIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const videoRefs = useRef<Array<HTMLVideoElement | null>>([]);
  const readyVideos = useRef(new Set<string>());
  const selectedCaseId = useRef(cases[0].id);
  const explorer = useRef<HTMLDivElement>(null);
  const manuallyPaused = useRef(false);
  const isVisible = useRef(false);
  const selectedCase = cases[caseIndex];

  const mediaPath = (
    kind: (typeof mediaKinds)[number],
    step: (typeof sampledSteps)[number]["assetStep"],
  ) => refinementMediaPath(basePath, selectedCase.id, kind, step, "mp4");

  const posterPath = (
    kind: (typeof mediaKinds)[number],
    step: (typeof sampledSteps)[number]["assetStep"],
  ) => refinementMediaPath(basePath, selectedCase.id, kind, step, "webp");

  const pauseTimeline = useCallback(() => {
    videoRefs.current.forEach((video) => video?.pause());
  }, []);

  const playTimeline = useCallback(() => {
    const videos = videoRefs.current.filter(
      (video): video is HTMLVideoElement => Boolean(video),
    );
    if (videos.length !== videoCount) return;

    void Promise.allSettled(videos.map((video) => video.play())).then(() => {
      setIsPlaying(!videos[0].paused);
    });
  }, []);

  const resetTimeline = useCallback(() => {
    videoRefs.current.forEach((video) => {
      if (video) video.currentTime = 0;
    });
  }, []);

  useEffect(() => {
    readyVideos.current.clear();
    pauseTimeline();
    resetTimeline();
    videoRefs.current.forEach((video) => video?.load());
  }, [caseIndex, pauseTimeline, resetTimeline]);

  useEffect(() => {
    const connection = (
      navigator as Navigator & {
        connection?: { saveData?: boolean; effectiveType?: string };
      }
    ).connection;

    if (
      connection?.saveData ||
      connection?.effectiveType === "slow-2g" ||
      connection?.effectiveType === "2g"
    ) {
      return;
    }

    const pendingPosters = cases.slice(1).flatMap((item) =>
      mediaKinds.flatMap((kind) =>
        sampledSteps.map(({ assetStep }) =>
          refinementMediaPath(basePath, item.id, kind, assetStep, "webp"),
        ),
      ),
    );
    let cancelled = false;
    let timeout = 0;
    let currentImage: HTMLImageElement | null = null;
    let nextPoster = 0;

    const preloadNextPoster = () => {
      if (cancelled || nextPoster >= pendingPosters.length) return;

      currentImage = new Image();
      currentImage.decoding = "async";
      currentImage.fetchPriority = "low";
      currentImage.onload = currentImage.onerror = () => {
        timeout = window.setTimeout(preloadNextPoster, 35);
      };
      currentImage.src = pendingPosters[nextPoster];
      nextPoster += 1;
    };

    timeout = window.setTimeout(preloadNextPoster, 900);
    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
      if (currentImage) {
        currentImage.onload = null;
        currentImage.onerror = null;
      }
    };
  }, [basePath]);

  useEffect(() => {
    const root = explorer.current;
    if (!root) return;

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    if (reducedMotion) {
      manuallyPaused.current = true;
      pauseTimeline();
      const frame = window.requestAnimationFrame(() => setIsPlaying(false));
      return () => window.cancelAnimationFrame(frame);
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        isVisible.current = entry.isIntersecting;
        if (!entry.isIntersecting) {
          pauseTimeline();
          setIsPlaying(false);
        } else if (
          !manuallyPaused.current &&
          readyVideos.current.size === videoCount
        ) {
          playTimeline();
        }
      },
      { threshold: 0.12 },
    );

    observer.observe(root);
    return () => {
      isVisible.current = false;
      observer.disconnect();
    };
  }, [pauseTimeline, playTimeline]);

  const markReady = (id: string) => {
    if (!id.startsWith(`${selectedCaseId.current}-`)) return;
    if (readyVideos.current.has(id)) return;
    readyVideos.current.add(id);
    setReadyVideoIds(new Set(readyVideos.current));

    if (
      readyVideos.current.size === videoCount &&
      !manuallyPaused.current &&
      isVisible.current
    ) {
      resetTimeline();
      playTimeline();
    }
  };

  const synchronizeTimeline = () => {
    const master = videoRefs.current[0];
    if (!master || !Number.isFinite(master.duration) || master.duration <= 0) {
      return;
    }

    const progress = master.currentTime / master.duration;
    videoRefs.current.slice(1).forEach((video) => {
      if (!video || !Number.isFinite(video.duration) || video.duration <= 0) {
        return;
      }

      const synchronizedTime = progress * video.duration;
      if (Math.abs(video.currentTime - synchronizedTime) > 0.12) {
        video.currentTime = synchronizedTime;
      }
    });
  };

  const restartTimeline = () => {
    resetTimeline();
    if (!manuallyPaused.current) playTimeline();
  };

  const togglePlayback = () => {
    const master = videoRefs.current[0];
    if (!master) return;

    if (master.paused) {
      manuallyPaused.current = false;
      playTimeline();
    } else {
      manuallyPaused.current = true;
      pauseTimeline();
      setIsPlaying(false);
    }
  };

  const selectCase = (index: number) => {
    if (index === caseIndex) return;

    pauseTimeline();
    resetTimeline();
    readyVideos.current.clear();
    selectedCaseId.current = cases[index].id;
    setReadyVideoIds(new Set());
    setIsPlaying(false);
    setCaseIndex(index);
  };

  const renderVideo = (
    kind: (typeof mediaKinds)[number],
    sampledStep: (typeof sampledSteps)[number],
    index: number,
  ) => {
    const { assetStep, displayStep } = sampledStep;
    const videoIndex = kind === "step" ? index : sampledSteps.length + index;
    const readyId = `${selectedCase.id}-${kind}-${assetStep}`;
    const isReady = readyVideoIds.has(readyId);
    const readableKind = kind === "step" ? "prediction" : "RMS delta-margin";

    return (
      <div
        className={`timeline-video${kind === "margin" ? " is-margin" : ""}${isReady ? " is-ready" : " is-loading"}`}
        key={readyId}
      >
        <video
          ref={(node) => {
            videoRefs.current[videoIndex] = node;
          }}
          muted
          playsInline
          preload="auto"
          poster={posterPath(kind, assetStep)}
          aria-label={`${selectedCase.label} ${readableKind} at refinement step ${displayStep}`}
          onCanPlay={() => markReady(readyId)}
          onTimeUpdate={videoIndex === 0 ? synchronizeTimeline : undefined}
          onEnded={videoIndex === 0 ? restartTimeline : undefined}
          onPlay={videoIndex === 0 ? () => setIsPlaying(true) : undefined}
          onPause={videoIndex === 0 ? () => setIsPlaying(false) : undefined}
        >
          <source src={mediaPath(kind, assetStep)} type="video/mp4" />
        </video>
        {!isReady && (
          <span className="timeline-loading-label" aria-hidden="true">
            Loading preview
          </span>
        )}
      </div>
    );
  };

  return (
    <div className="refinement-explorer" ref={explorer}>
      <div className="case-switcher">
        <div className="case-switcher-heading">
          <p>
            <span>Explore the trajectories</span>
            <strong>Choose an editing type</strong>
          </p>
          <small>7 cases · every sampled step updates together</small>
        </div>

        <div className="case-tabs" aria-label="Choose an editing case">
          {cases.map((item, index) => (
            <button
              type="button"
              className={index === caseIndex ? "is-active" : undefined}
              aria-pressed={index === caseIndex}
              onClick={() => selectCase(index)}
              key={item.id}
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="explorer-meta">
        <p>
          <span>Selected instruction</span>
          “{selectedCase.prompt}”
        </p>
        <div className="trajectory-range">
          <span>Full sampled trajectory</span>
          <strong>Step 1 → Step 50</strong>
        </div>
      </div>

      <div
        className="timeline-scroll"
        role="region"
        aria-label={`${selectedCase.label} refinement trajectory from step 1 to step 50`}
      >
        <div className="timeline-grid">
          <div className="timeline-corner">
            <span>Refinement state</span>
          </div>
          {sampledSteps.map(({ assetStep, displayStep }) => (
            <div className="timeline-step" key={`header-${assetStep}`}>
              <span>Step</span>
              <strong>{displayStep}</strong>
            </div>
          ))}

          <div className="timeline-row-label">
            <span>01</span>
            <strong>Prediction</strong>
            <small>Pixel space</small>
          </div>
          {sampledSteps.map((sampledStep, index) =>
            renderVideo("step", sampledStep, index),
          )}

          <div className="timeline-row-label margin-row-label">
            <span>02</span>
            <strong>RMS Δ-Margin</strong>
            <small>VAE latent space</small>
            <i aria-hidden="true" />
          </div>
          {sampledSteps.map((sampledStep, index) =>
            renderVideo("margin", sampledStep, index),
          )}
        </div>
      </div>

      <div className="explorer-footer">
        <p>
          Each column pairs the prediction with the RMS Δ-Margin from the same
          refinement step. All twelve videos share normalized time. Margin maps
          are enlarged proportionally from VAE latent space; brighter regions
          indicate a larger leave-one-branch-out change in bit margin.
        </p>
        <button
          type="button"
          onClick={togglePlayback}
          disabled={readyVideoIds.size < videoCount}
        >
          <span className={isPlaying ? "pause-icon" : "play-icon"} aria-hidden="true" />
          <span aria-live="polite">
            {readyVideoIds.size < videoCount
              ? `Loading ${readyVideoIds.size} / ${videoCount}`
              : isPlaying
                ? "Pause timeline"
                : "Play timeline"}
          </span>
        </button>
      </div>
    </div>
  );
}
