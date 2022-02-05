export const getCorrelation = (x: number[], y: number[], returnDecimals: number): number => {
  const mean = { x: calculateAverage(x), y: calculateAverage(y) };
  const std = { x: calculateStdDev(x), y: calculateStdDev(y) };

  const addedMultipliedDifferences = x.map((val, i) => (val - mean.x) * (y[i] - mean.y)).reduce((sum, v) => sum + v, 0);

  const dividedByDevs = addedMultipliedDifferences / (std.x * std.y);

  const r = dividedByDevs / (x.length - 1);
  return preciseRound(r, returnDecimals);
};

const calculateAverage = (values: number[]): number => {
  return values.reduce((sum, v) => sum + v, 0) / values.length;
};

const calculateStdDev = (values: number[]): number => {
  const µ = calculateAverage(values);
  const addedSquareDiffs = values
    .map(val => val - µ)
    .map(diff => diff ** 2)
    .reduce((sum, v) => sum + v, 0);
  const variance = addedSquareDiffs / (values.length - 1);
  return Math.sqrt(variance);
};

const preciseRound = (num: number, dec: number) =>
  Math.round(num * 10 ** dec + (num >= 0 ? 1 : -1) * 0.0001) / 10 ** dec;
