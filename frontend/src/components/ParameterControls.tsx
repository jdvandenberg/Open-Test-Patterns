import type { Parameter, ParamValues } from "../types";

interface Props {
  parameters: Parameter[];
  values: ParamValues;
  onChange: (name: string, value: unknown) => void;
}

function toColorHex(rgb: number[]): string {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));
  return (
    "#" +
    [rgb[0], rgb[1], rgb[2]]
      .map((c) => clamp(c ?? 0).toString(16).padStart(2, "0"))
      .join("")
  );
}

function fromColorHex(hex: string): number[] {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [r, g, b];
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
              <div className="color-row">
                <input
                  type="color"
                  disabled={disabled}
                  value={toColorHex(value as number[])}
                  onChange={(e) => onChange(p.name, fromColorHex(e.target.value))}
                />
                {(value as number[]).map((c, i) => (
                  <input
                    key={i}
                    type="number"
                    step={0.001}
                    min={0}
                    disabled={disabled}
                    value={Number(c)}
                    onChange={(e) => {
                      const next = [...(value as number[])];
                      next[i] = Number(e.target.value);
                      onChange(p.name, next);
                    }}
                  />
                ))}
              </div>
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
