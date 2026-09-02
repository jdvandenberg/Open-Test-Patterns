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
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
