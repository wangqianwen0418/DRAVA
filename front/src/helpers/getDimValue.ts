import { TResultRow } from 'types';

export const getDimValues = (samples: TResultRow[], dimName: string): number[] => {
  return samples.map(s => parseFloat(s[dimName]) || s[dimName]);
};
