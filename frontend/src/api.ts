import type { ImageFormat, ParamValues, Pattern } from "./types";

const JSON_HEADERS = { "Content-Type": "application/json" };

export async function fetchPatterns(): Promise<Pattern[]> {
  const res = await fetch("/api/patterns");
  if (!res.ok) throw new Error("Failed to load patterns");
  return res.json();
}

export async function fetchFormats(): Promise<ImageFormat[]> {
  const res = await fetch("/api/formats");
  if (!res.ok) throw new Error("Failed to load formats");
  return res.json();
}

export interface RenderSpec {
  pattern_id: string;
  width: number;
  height: number;
  params: ParamValues;
}

export function downloadFilename(
  name: string,
  width: number,
  height: number,
  extension: string,
): string {
  const stem = name.replace(/[^\w]+/g, "_").replace(/^_|_$/g, "");
  return `${stem}_${width}x${height}.${extension}`;
}

export async function fetchPreview(spec: RenderSpec, signal?: AbortSignal): Promise<Blob> {
  const res = await fetch("/api/preview", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(spec),
    signal,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? "Preview failed");
  }
  return res.blob();
}

export async function downloadRender(
  spec: RenderSpec & {
    format: string;
    bit_depth: number | null;
    compression?: string | null;
    filename: string;
  },
): Promise<void> {
  const { filename, ...body } = spec;
  const res = await fetch("/api/render", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? "Render failed");
  }
  const blob = await res.blob();
  triggerDownload(blob, filename);
}

export interface ToneSpec {
  frequency: number;
  duration: number;
  loudness: number;
  waveform: string;
  frequency_low: number;
  frequency_high: number;
}

function noiseBand(spec: ToneSpec): { lo: number; hi: number } {
  const lo = Math.min(spec.frequency_low, spec.frequency_high);
  const hi = Math.max(spec.frequency_low, spec.frequency_high);
  return { lo, hi };
}

export function toneFilename(spec: ToneSpec): string {
  const kind = spec.waveform.charAt(0).toUpperCase() + spec.waveform.slice(1);
  if (spec.waveform === "white" || spec.waveform === "pink") {
    const { lo, hi } = noiseBand(spec);
    return `${kind}_Noise_${lo}-${hi}Hz_${spec.duration}s_${spec.loudness}dBFS.wav`;
  }
  return `${kind}_${spec.frequency}Hz_${spec.duration}s_${spec.loudness}dBFS.wav`;
}

export async function fetchTone(spec: ToneSpec, signal?: AbortSignal): Promise<Blob> {
  const res = await fetch("/api/tone", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(spec),
    signal,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? "Tone failed");
  }
  return res.blob();
}

export async function downloadTone(spec: ToneSpec): Promise<void> {
  const blob = await fetchTone(spec);
  triggerDownload(blob, toneFilename(spec));
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
