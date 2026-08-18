"use client";

import { useEffect, useRef, useState } from "react";

type ResultVideoProps = {
  src: string;
  poster: string;
  label: string;
  featured?: boolean;
};

export function ResultVideo({
  src,
  poster,
  label,
  featured = false,
}: ResultVideoProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const manuallyPaused = useRef(false);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isMuted, setIsMuted] = useState(true);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    if (reducedMotion) {
      manuallyPaused.current = true;
      video.pause();
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) {
          video.pause();
          setIsPlaying(false);
          return;
        }

        if (!manuallyPaused.current) {
          void video.play().catch(() => setIsPlaying(false));
        }
      },
      { threshold: 0.2 },
    );

    observer.observe(video);
    return () => observer.disconnect();
  }, []);

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      manuallyPaused.current = false;
      void video.play().catch(() => setIsPlaying(false));
    } else {
      manuallyPaused.current = true;
      video.pause();
    }
  };

  const toggleMuted = () => {
    const video = videoRef.current;
    if (!video) return;
    const nextMuted = !video.muted;
    video.muted = nextMuted;
    setIsMuted(nextMuted);
  };

  return (
    <div className={`result-video${featured ? " is-featured" : ""}`}>
      <video
        ref={videoRef}
        autoPlay
        loop
        muted
        playsInline
        preload={featured ? "auto" : "metadata"}
        poster={poster}
        aria-label={label}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      >
        <source src={src} type="video/mp4" />
        Your browser does not support MP4 video playback.
      </video>
      <div className="video-controls">
        <button
          type="button"
          onClick={togglePlayback}
          aria-label={isPlaying ? `Pause ${label}` : `Play ${label}`}
          title={isPlaying ? "Pause" : "Play"}
        >
          <span className={isPlaying ? "pause-icon" : "play-icon"} aria-hidden="true" />
        </button>
        <button
          type="button"
          onClick={toggleMuted}
          aria-label={isMuted ? `Unmute ${label}` : `Mute ${label}`}
          title={isMuted ? "Unmute" : "Mute"}
        >
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="M3.5 8v4h3l4 3V5l-4 3h-3Z" />
            {isMuted ? (
              <path d="m13.5 8 3 4m0-4-3 4" />
            ) : (
              <path d="M13.5 7.2c1.4 1.55 1.4 4.05 0 5.6" />
            )}
          </svg>
        </button>
      </div>
    </div>
  );
}
