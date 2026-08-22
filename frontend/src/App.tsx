import { useEffect, useMemo, useRef, useState } from "react";
import { downloadRender, fetchFormats, fetchPatterns, fetchPreview } from "./api";
import { ParameterControls } from "./components/ParameterControls";
import type { ImageFormat, ParamValues, Pattern } from "./types";

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
    if (!currentFormat) return;
    setBitDepth(currentFormat.default_bit_depth);
    setCompression(currentFormat.default_compression);
  }, [currentFormat]);

  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!selected) return;
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
  }, [selected, params, width, height]);

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
      await downloadRender({
        pattern_id: selected.id,
        width,
        height,
        params,
        format: formatId,
        bit_depth: bitDepth,
        compression: currentFormat?.compressions.length ? compression : null,
      });
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
      </aside>

      <main className="stage">
        <div className="preview-wrap">
          {previewUrl ? (
            <img className="preview" src={previewUrl} alt={selected?.name ?? "preview"} />
          ) : (
            <div className="preview placeholder">Select a pattern</div>
          )}
          {loading && <div className="loading-badge">Rendering…</div>}
        </div>
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

            <h2>Output</h2>
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

            <button className="download" onClick={onDownload} disabled={downloading}>
              {downloading ? "Rendering…" : `Download .${currentFormat?.extension ?? formatId}`}
            </button>
          </>
        )}
      </section>
    </div>
  );
}
