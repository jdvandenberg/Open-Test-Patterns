import { useEffect, useMemo, useRef, useState } from "react";
import { downloadFilename, downloadRender, downloadTone, fetchFormats, fetchPatterns, fetchPreview, fetchTone } from "./api";
import { ParameterControls } from "./components/ParameterControls";
import { PatternPreview } from "./components/PatternPreview";
import type { ImageFormat, ParamValues, Pattern } from "./types";

const ASPECT_RATIO_ID = "aspect-ratio";
const ASPECT_RATIO_CONTAINERS: Record<string, { w: number; h: number }> = {
  "dci-2k-full": { w: 2048, h: 1080 },
  "dci-2k-flat": { w: 1998, h: 1080 },
  "dci-2k-scope": { w: 2048, h: 858 },
  "dci-4k-full": { w: 4096, h: 2160 },
  "dci-4k-flat": { w: 3996, h: 2160 },
  "dci-4k-scope": { w: 4096, h: 1716 },
  "uhd-4k": { w: 3840, h: 2160 },
  "dci-8k": { w: 8192, h: 4320 },
  "uhd-8k": { w: 7680, h: 4320 },
  "1080p": { w: 1920, h: 1080 },
};

const RESOLUTIONS: { label: string; w: number; h: number }[] = [
  { label: "HD 1920×1080", w: 1920, h: 1080 },
  { label: "UHD 3840×2160", w: 3840, h: 2160 },
  { label: "DCI 2K 2048×1080", w: 2048, h: 1080 },
  { label: "DCI 4K 4096×2160", w: 4096, h: 2160 },
  { label: "8K 7680×4320", w: 7680, h: 4320 },
  { label: "Square 2048×2048", w: 2048, h: 2048 },
];

function defaultParams(pattern: Pattern): ParamValues {
  const values: ParamValues = {};
  for (const p of pattern.parameters) values[p.name] = p.default;
  return values;
}

function toneSpec(params: ParamValues) {
  return {
    waveform: String(params.waveform ?? "sine"),
    frequency: Number(params.frequency ?? 440),
    frequency_low: Number(params.frequency_low ?? 20),
    frequency_high: Number(params.frequency_high ?? 20_000),
    duration: Number(params.duration ?? 5),
    loudness: Number(params.loudness ?? -20),
    sample_rate: Number(params.sample_rate ?? 44100),
    bit_depth: String(params.bit_depth ?? "16"),
  };
}

function formatHertz(hz: number): string {
  if (hz >= 1000) {
    const khz = hz / 1000;
    const text = Number.isInteger(khz) ? String(khz) : khz.toPrecision(4).replace(/\.?0+$/, "");
    return `${text} kHz`;
  }
  return `${hz} Hz`;
}

function toneFormatLabel(params: ParamValues): string {
  const spec = toneSpec(params);
  const rate = spec.sample_rate === 44100 ? "44.1 kHz" : `${spec.sample_rate / 1000} kHz`;
  const depth = spec.bit_depth === "float32" ? "32-bit float" : `${spec.bit_depth}-bit`;
  return `${spec.duration} s · ${spec.loudness} dBFS · ${rate} · ${depth}`;
}

function toneHeadline(params: ParamValues): string {
  const spec = toneSpec(params);
  if (spec.waveform === "white" || spec.waveform === "pink" || spec.waveform === "sweep") {
    const lo = spec.waveform === "sweep" ? spec.frequency_low : Math.min(spec.frequency_low, spec.frequency_high);
    const hi = spec.waveform === "sweep" ? spec.frequency_high : Math.max(spec.frequency_low, spec.frequency_high);
    const kind =
      spec.waveform === "white" ? "White noise" : spec.waveform === "pink" ? "Pink noise" : "Sweep";
    return `${kind} ${formatHertz(lo)} – ${formatHertz(hi)}`;
  }
  const labels: Record<string, string> = {
    sine: "Sine",
    triangle: "Triangle",
    sawtooth: "Sawtooth",
    square: "Square",
  };
  const name = labels[spec.waveform] ?? spec.waveform;
  return `${name} ${formatHertz(spec.frequency)}`;
}

export function App() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [formats, setFormats] = useState<ImageFormat[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [params, setParams] = useState<ParamValues>({});
  const [search, setSearch] = useState("");
  const [width, setWidth] = useState(1920);
  const [height, setHeight] = useState(1080);
  const [formatId, setFormatId] = useState("exr");
  const [bitDepth, setBitDepth] = useState<number | null>(null);
  const [compression, setCompression] = useState<string | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  const selected = useMemo(
    () => patterns.find((p) => p.id === selectedId) ?? null,
    [patterns, selectedId],
  );
  const isAudio = selected?.kind === "audio";
  const isAspectRatio = selected?.id === ASPECT_RATIO_ID;
  const currentFormat = useMemo(
    () => formats.find((f) => f.id === formatId) ?? null,
    [formats, formatId],
  );

  useEffect(() => {
    fetchPatterns()
      .then((data) => {
        setPatterns(data);
        if (data.length) {
          setSelectedId(data.find((p) => p.id === "smpte-rp219-2-2016")?.id ?? data[0].id);
        }
      })
      .catch((e) => setError(String(e)));
    fetchFormats().then(setFormats).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (selected) setParams(defaultParams(selected));
  }, [selected]);

  useEffect(() => {
    if (!isAspectRatio) return;
    const size = ASPECT_RATIO_CONTAINERS[String(params.container ?? "1080p")];
    if (!size) return;
    setWidth(size.w);
    setHeight(size.h);
  }, [isAspectRatio, params.container]);

  useEffect(() => {
    if (!currentFormat) return;
    setBitDepth(currentFormat.default_bit_depth);
    setCompression(currentFormat.default_compression);
  }, [currentFormat]);

  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!selected || isAudio) return;
    const handle = setTimeout(() => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      setError(null);
      fetchPreview(
        { pattern_id: selected.id, width, height, params },
        controller.signal,
      )
        .then((blob) => {
          setPreviewUrl((old) => {
            if (old) URL.revokeObjectURL(old);
            return URL.createObjectURL(blob);
          });
        })
        .catch((e) => {
          if (e.name !== "AbortError") setError(String(e.message ?? e));
        })
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(handle);
  }, [selected, params, width, height, isAudio]);

  useEffect(() => {
    if (!selected || !isAudio) return;
    const handle = setTimeout(() => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      setError(null);
      fetchTone({ ...toneSpec(params), bit_depth: "16" }, controller.signal)
        .then((blob) => {
          setPreviewUrl((old) => {
            if (old) URL.revokeObjectURL(old);
            return URL.createObjectURL(blob);
          });
        })
        .catch((e) => {
          if (e.name !== "AbortError") setError(String(e.message ?? e));
        })
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(handle);
  }, [selected, params, isAudio]);

  const grouped = useMemo(() => {
    const filtered = patterns.filter(
      (p) =>
        p.name.toLowerCase().includes(search.toLowerCase()) ||
        p.category.toLowerCase().includes(search.toLowerCase()),
    );
    const map = new Map<string, Pattern[]>();
    for (const p of filtered) {
      if (!map.has(p.category)) map.set(p.category, []);
      map.get(p.category)!.push(p);
    }
    return [...map.entries()];
  }, [patterns, search]);

  const onParamChange = (name: string, value: unknown) =>
    setParams((prev) => ({ ...prev, [name]: value }));

  const onDownload = async () => {
    if (!selected) return;
    setDownloading(true);
    setError(null);
    try {
      if (isAudio) {
        await downloadTone(toneSpec(params));
      } else {
        await downloadRender({
          pattern_id: selected.id,
          width,
          height,
          params,
          format: formatId,
          bit_depth: bitDepth,
          compression: currentFormat?.compressions.length ? compression : null,
          filename: downloadFilename(
            selected.name,
            width,
            height,
            currentFormat?.extension ?? formatId,
          ),
        });
      }
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <h1>Open Test Patterns</h1>
          <p>Color-accurate charts for display calibration &amp; HDR QC</p>
        </div>
        <input
          className="search"
          placeholder="Search patterns…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <nav className="pattern-list">
          {grouped.map(([category, items]) => (
            <div key={category} className="pattern-group">
              <h3>{category}</h3>
              {items.map((p) => (
                <button
                  key={p.id}
                  className={p.id === selectedId ? "pattern-item active" : "pattern-item"}
                  onClick={() => setSelectedId(p.id)}
                >
                  {p.name}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <a
          className="github-link"
          href="https://github.com/jdvandenberg/Open-Test-Patterns"
          target="_blank"
          rel="noopener noreferrer"
        >
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8" />
          </svg>
          GitHub
        </a>
      </aside>

      <main className="stage">
        {isAudio ? (
          <div className="preview-wrap">
            {previewUrl ? (
              <div className="audio-preview">
                <div className="audio-hz">{toneHeadline(params)}</div>
                <div className="audio-meta">
                  {toneFormatLabel(params)}
                </div>
                <audio key={previewUrl} controls src={previewUrl} />
              </div>
            ) : (
              <div className="preview placeholder">Generating tone…</div>
            )}
            {loading && <div className="loading-badge">Generating…</div>}
          </div>
        ) : previewUrl ? (
            <PatternPreview
              src={previewUrl}
              alt={selected?.name ?? "preview"}
              resetKey={selectedId}
              loading={loading}
            />
        ) : (
          <div className="preview-wrap">
            <div className="preview placeholder">Select a pattern</div>
            {loading && <div className="loading-badge">Rendering…</div>}
          </div>
        )}
        {selected && (
          <div className="stage-caption">
            <strong>{selected.name}</strong> — {selected.description}
          </div>
        )}
        {error && <div className="error-bar">{error}</div>}
      </main>

      <section className="inspector">
        {selected && (
          <>
            <h2>Parameters</h2>
            <ParameterControls
              parameters={selected.parameters}
              values={params}
              onChange={onParamChange}
            />

            {!isAudio && (
              <>
                <h2>Output</h2>
                {!isAspectRatio && (
                  <>
                    <div className="field">
                      <label>Resolution</label>
                      <select
                        value={`${width}x${height}`}
                        onChange={(e) => {
                          const preset = RESOLUTIONS.find((r) => `${r.w}x${r.h}` === e.target.value);
                          if (preset) {
                            setWidth(preset.w);
                            setHeight(preset.h);
                          }
                        }}
                      >
                        {RESOLUTIONS.map((r) => (
                          <option key={r.label} value={`${r.w}x${r.h}`}>
                            {r.label}
                          </option>
                        ))}
                        {!RESOLUTIONS.some((r) => r.w === width && r.h === height) && (
                          <option value={`${width}x${height}`}>
                            Custom {width}×{height}
                          </option>
                        )}
                      </select>
                    </div>
                    <div className="field range-row">
                      <input
                        type="number"
                        value={width}
                        min={1}
                        max={16384}
                        onChange={(e) => setWidth(Number(e.target.value))}
                      />
                      <span className="times">×</span>
                      <input
                        type="number"
                        value={height}
                        min={1}
                        max={16384}
                        onChange={(e) => setHeight(Number(e.target.value))}
                      />
                    </div>
                  </>
                )}

                <div className="field">
                  <label>Format</label>
                  <select value={formatId} onChange={(e) => setFormatId(e.target.value)}>
                    {formats.map((f) => (
                      <option key={f.id} value={f.id}>
                        {f.label}
                      </option>
                    ))}
                  </select>
                </div>

                {currentFormat && currentFormat.allowed_bit_depths.length > 1 && (
                  <div className="field">
                    <label>Bit depth</label>
                    <select
                      value={bitDepth ?? currentFormat.default_bit_depth}
                      onChange={(e) => setBitDepth(Number(e.target.value))}
                    >
                      {currentFormat.allowed_bit_depths.map((d) => (
                        <option key={d} value={d}>
                          {d === 16 && currentFormat.id === "exr" ? "16 (half)" : `${d}-bit`}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {currentFormat && currentFormat.compressions.length > 0 && (
                  <div className="field">
                    <label>Compression</label>
                    <select
                      value={compression ?? currentFormat.default_compression ?? ""}
                      onChange={(e) => setCompression(e.target.value)}
                    >
                      {currentFormat.compressions.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.label}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </>
            )}

            <button className="download" onClick={onDownload} disabled={downloading}>
              {downloading
                ? isAudio
                  ? "Generating…"
                  : "Rendering…"
                : isAudio
                  ? "Download .wav"
                  : `Download .${currentFormat?.extension ?? formatId}`}
            </button>
          </>
        )}
      </section>
    </div>
  );
}
