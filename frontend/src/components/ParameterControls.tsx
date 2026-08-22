import type { Parameter, ParamValues } from "../types";

interface Props {
  parameters: Parameter[];
  values: ParamValues;
  onChange: (name: string, value: unknown) => void;
}

// A color control's components are not always [0, 1]: a 12-bit code-value input
// declares [0, 4095]. Everything below is driven by the declared range so one
// control serves both, and adding another depth needs no frontend change.
interface ColorScale {
  lo: number;
  hi: number;
  step: number;
  integral: boolean;
}

function colorScale(p: Parameter): ColorScale {
  const lo = p.minimum ?? 0;
  const hi = p.maximum ?? 1;
  const step = p.step ?? 0.001;
  return { lo, hi, step, integral: step >= 1 };
}

function toColorHex(rgb: number[], scale: ColorScale): string {
  const to8bit = (v: number) => {
    const unit = (clampComponent(v ?? 0, scale) - scale.lo) / (scale.hi - scale.lo || 1);
    return Math.max(0, Math.min(255, Math.round(unit * 255)));
  };
  return (
    "#" +
    [rgb[0], rgb[1], rgb[2]].map((c) => to8bit(c).toString(16).padStart(2, "0")).join("")
  );
}

// Mirrors the clamp the backend applies to color components, so the field shows
// the value that will actually be rendered rather than what was typed.
function clampComponent(v: number, scale: ColorScale): number {
  if (!Number.isFinite(v)) return scale.lo;
  const clamped = Math.max(scale.lo, Math.min(scale.hi, v));
  return scale.integral ? Math.round(clamped) : clamped;
}

function fromColorHex(hex: string, scale: ColorScale): number[] {
  const span = scale.hi - scale.lo;
  const channel = (start: number) => {
    const unit = parseInt(hex.slice(start, start + 2), 16) / 255;
    return clampComponent(scale.lo + unit * span, scale);
  };
  return [channel(1), channel(3), channel(5)];
}

interface ColorRowProps {
  p: Parameter;
  value: number[];
  disabled: boolean;
  onChange: (name: string, value: unknown) => void;
}

function ColorRow({ p, value, disabled, onChange }: ColorRowProps) {
  const scale = colorScale(p);
  return (
    <div className="color-row">
      <input
        type="color"
        disabled={disabled}
        value={toColorHex(value, scale)}
        onChange={(e) => onChange(p.name, fromColorHex(e.target.value, scale))}
      />
      {value.map((c, i) => (
        <input
          key={i}
          type="number"
          step={scale.step}
          min={scale.lo}
          max={scale.hi}
          disabled={disabled}
          value={Number(c)}
          onChange={(e) => {
            const next = [...value];
            next[i] = clampComponent(Number(e.target.value), scale);
            onChange(p.name, next);
          }}
        />
      ))}
      <span className="range-note">
        [{scale.lo}, {scale.hi}]
      </span>
    </div>
  );
}

export function ParameterControls({ parameters, values, onChange }: Props) {
  const resolve = (name: string): unknown => {
    const v = values[name];
    return v !== undefined ? v : parameters.find((q) => q.name === name)?.default;
  };

  return (
    <div className="params">
      {parameters.map((p) => {
        const value = values[p.name] ?? p.default;
        const dep = p.disabled_when;
        const disabled = dep ? dep.values.includes(String(resolve(dep.parameter))) : false;
        return (
          <div className={disabled ? "field disabled" : "field"} key={p.name}>
            <label title={disabled ? dep!.reason || p.description : p.description}>
              {p.label}
              {p.unit ? <span className="unit"> ({p.unit})</span> : null}
            </label>

            {p.type === "bool" && (
              <input
                type="checkbox"
                disabled={disabled}
                checked={Boolean(value)}
                onChange={(e) => onChange(p.name, e.target.checked)}
              />
            )}

            {p.type === "choice" && (
              <select
                value={String(value)}
                disabled={disabled}
                onChange={(e) => onChange(p.name, e.target.value)}
              >
                {p.choices.map((c) => (
                  <option key={c.value} value={c.value}>
                    {c.label}
                  </option>
                ))}
              </select>
            )}

            {(p.type === "int" || p.type === "float") && (
              <div className="range-row">
                {p.minimum !== null && p.maximum !== null && (
                  <input
                    type="range"
                    min={p.minimum}
                    max={p.maximum}
                    step={p.step ?? (p.type === "int" ? 1 : 0.01)}
                    disabled={disabled}
                    value={Number(value)}
                    onChange={(e) => onChange(p.name, Number(e.target.value))}
                  />
                )}
                <input
                  type="number"
                  min={p.minimum ?? undefined}
                  max={p.maximum ?? undefined}
                  step={p.step ?? (p.type === "int" ? 1 : 0.01)}
                  disabled={disabled}
                  value={Number(value)}
                  onChange={(e) => onChange(p.name, Number(e.target.value))}
                />
              </div>
            )}

            {p.type === "color" && (
              <ColorRow p={p} value={value as number[]} disabled={disabled} onChange={onChange} />
            )}

            {disabled && dep!.reason ? (
              <p className="hint">{dep!.reason}</p>
            ) : (
              p.description && <p className="hint">{p.description}</p>
            )}
          </div>
        );
      })}
    </div>
  );
}
