export type ParamType = "int" | "float" | "bool" | "choice" | "color";

export interface Choice {
  value: string;
  label: string;
}

export interface DisabledWhen {
  parameter: string;
  values: string[];
  reason: string;
}

export interface Parameter {
  name: string;
  label: string;
  type: ParamType;
  default: unknown;
  minimum: number | null;
  maximum: number | null;
  step: number | null;
  choices: Choice[];
  unit: string | null;
  description: string;
  disabled_when: DisabledWhen | null;
}

export interface Pattern {
  id: string;
  name: string;
  category: string;
  description: string;
  parameters: Parameter[];
}

export interface Compression {
  id: string;
  label: string;
  lossless: boolean;
}

export interface ImageFormat {
  id: string;
  label: string;
  extension: string;
  default_bit_depth: number;
  allowed_bit_depths: number[];
  compressions: Compression[];
  default_compression: string | null;
}

export type ParamValues = Record<string, unknown>;
