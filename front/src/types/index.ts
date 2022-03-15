export type TCSVResultRow = {
  chr: number;
  start: number;
  end: number;
  z: string; // '[m, m, m,]',
  std: string; // '[std, std, std,]',
  [key: string]: any;
};

export type TResultRow = {
  id: string; // id is the sample index in the whole dataset
  index: number; // index is the sample index in all shown data
  chr: number;
  start: number;
  end: number;
  z: number[]; // '[x, x, x,]',
  assignments: { [dimName: string]: number }; // index of the assigned groups at each dim
  [key: string]: any;
};

export type TDistribution = {
  histogram: number[];
  groupedSamples: string[][]; // sample id grouped
  labels: string[]; // name of each group
};

export type TFilter = { [dimName: string]: boolean[] };
interface State {
  dataset: string;
  filters: TFilter;
  samples: TResultRow[];
}

export type TMatrixData = { [dimName: string]: TDistribution };
