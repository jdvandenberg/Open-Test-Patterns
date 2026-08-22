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
  },
): Promise<void> {
  const res = await fetch("/api/render", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(spec),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? "Render failed");
  }
  const blob = await res.blob();
  const disposition = res.headers.get("content-disposition") ?? "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match ? match[1] : `${spec.pattern_id}.${spec.format}`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
