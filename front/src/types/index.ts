export type TCSVResultRow = {
  chr: number;
  start: number;
  end: number;
  z: string; // '[x, x, x,]',
  [key: string]: any;
};

export type TResultRow = {
  id: string;
  chr: number;
  start: number;
  end: number;
  z: number[]; // '[x, x, x,]',
  [key: string]: any;
};

export type TDistribution = {
  histogram: number[];
  groupedSamples: string[][]; // sample id grouped
  labels: string[]; // name of each group
};
