import { TResultRow } from 'types';

export const getDimValues = (samples: TResultRow[], dimName: string): number[] => {
  if (dimName.includes('dim')) {
    const dimNum = parseInt(dimName.split('_')[1]);
    return samples.map(sample => sample['z'][dimNum]);
  } else if (dimName == 'size') {
    return samples.map(s => s.end - s.start);
  } else {
    return samples.map(s => parseFloat(s[dimName]) || s[dimName]);
  }
};
