import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 16;
const ZOOM_FACTOR = 1.25;
const VIEW_PAD = 48;

function clampZoom(zoom: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
}

function drawNearest(
  source: HTMLImageElement,
  canvas: HTMLCanvasElement,
  destW: number,
  destH: number,
): void {
  const dpr = window.devicePixelRatio || 1;
  const outW = Math.max(1, Math.round(destW * dpr));
  const outH = Math.max(1, Math.round(destH * dpr));
  canvas.style.width = `${destW}px`;
  canvas.style.height = `${destH}px`;
  canvas.width = outW;
  canvas.height = outH;

  // Copy 1:1 first so the scaled blit samples original pixels, not a
  // browser-smoothed decoded bitmap.
  const native = document.createElement("canvas");
  native.width = source.naturalWidth;
  native.height = source.naturalHeight;
  const nativeCtx = native.getContext("2d");
  const ctx = canvas.getContext("2d");
  if (!nativeCtx || !ctx) return;
  nativeCtx.drawImage(source, 0, 0);
  ctx.imageSmoothingEnabled = false;
  (
    ctx as CanvasRenderingContext2D & { webkitImageSmoothingEnabled?: boolean }
  ).webkitImageSmoothingEnabled = false;
  ctx.drawImage(native, 0, 0, outW, outH);
}

interface Props {
  src: string;
  alt: string;
  resetKey?: string | null;
  loading?: boolean;
}

export function PatternPreview({ src, alt, resetKey, loading }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<{ x: number; y: number; sl: number; st: number } | null>(null);
  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [fit, setFit] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [panning, setPanning] = useState(false);

  useEffect(() => {
    setZoom(1);
    const el = wrapRef.current;
    if (el) {
      el.scrollLeft = 0;
      el.scrollTop = 0;
    }
  }, [resetKey]);

  const measureFit = useCallback(() => {
    const el = wrapRef.current;
    if (!el || !natural.w || !natural.h) return;
    const availW = Math.max(1, el.clientWidth - VIEW_PAD);
    const availH = Math.max(1, el.clientHeight - VIEW_PAD);
    setFit(Math.min(1, availW / natural.w, availH / natural.h));
  }, [natural]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => measureFit());
    ro.observe(el);
    measureFit();
    return () => ro.disconnect();
  }, [measureFit]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
      setZoom((z) => clampZoom(z * factor));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const displayW = natural.w ? Math.max(1, Math.round(natural.w * fit * zoom)) : 0;
  const displayH = natural.h ? Math.max(1, Math.round(natural.h * fit * zoom)) : 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !img.naturalWidth || !displayW) return;
    drawNearest(img, canvas, displayW, displayH);
  }, [src, natural, displayW, displayH]);

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const el = wrapRef.current;
    if (!el) return;
    dragRef.current = { x: e.clientX, y: e.clientY, sl: el.scrollLeft, st: el.scrollTop };
    el.setPointerCapture(e.pointerId);
    setPanning(true);
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    const el = wrapRef.current;
    if (!drag || !el) return;
    el.scrollLeft = drag.sl - (e.clientX - drag.x);
    el.scrollTop = drag.st - (e.clientY - drag.y);
  };

  const endPan = (e?: PointerEvent<HTMLDivElement>) => {
    const el = wrapRef.current;
    if (e && el && el.hasPointerCapture(e.pointerId)) {
      el.releasePointerCapture(e.pointerId);
    }
    dragRef.current = null;
    setPanning(false);
  };

  const zoomIn = () => setZoom((z) => clampZoom(z * ZOOM_FACTOR));
  const zoomOut = () => setZoom((z) => clampZoom(z / ZOOM_FACTOR));
  const zoomFit = () => setZoom(1);
  const zoomActual = () => {
    if (fit > 0) setZoom(clampZoom(1 / fit));
  };

  return (
    <div className="preview-stage">
      <div
        ref={wrapRef}
        className={panning ? "preview-wrap is-panning" : "preview-wrap"}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endPan}
        onPointerCancel={endPan}
      >
        <img
          ref={imageRef}
          className="preview-measure"
          src={src}
          alt={alt}
          draggable={false}
          onLoad={(e) => {
            const img = e.currentTarget;
            setNatural({ w: img.naturalWidth, h: img.naturalHeight });
          }}
        />
        <canvas ref={canvasRef} className="preview" />
      </div>
      <div className="zoom-controls">
        <button type="button" onClick={zoomOut} title="Zoom out" aria-label="Zoom out">
          −
        </button>
        <button type="button" className="zoom-level" onClick={zoomFit} title="Fit to view">
          {zoom === 1 ? "Fit" : `${Math.round(zoom * 100)}%`}
        </button>
        <button type="button" onClick={zoomIn} title="Zoom in" aria-label="Zoom in">
          +
        </button>
        <button type="button" className="zoom-actual" onClick={zoomActual} title="Actual pixels (1:1)">
          1:1
        </button>
      </div>
      {loading && <div className="loading-badge">Rendering…</div>}
    </div>
  );
}
