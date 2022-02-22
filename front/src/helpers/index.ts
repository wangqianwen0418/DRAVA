export * from './hist';
export * from './getDimValue';
export * from './getCorrelation';
export { debounce } from './debounce';

export const getSum = (arr: number[]): number => {
  return arr.reduce((a, b) => a + b, 0);
};

export const getAbsSum = (arr: number[]): number => {
  return arr.map(d => Math.abs(d)).reduce((a, b) => a + b, 0);
};
