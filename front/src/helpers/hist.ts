import { STEP_NUM, RANGE_MAX, RANGE_MIN } from 'Const';

/**
 * similar to the python range function
 * @param v : length of array
 * @param fill:
 * - when given, fill the array with this constant value
 * - otherwise, [0, 1, .., v-1]
 * @returns array
 */
export const range = (v: number, fill?: number): number[] => {
  return Array.from(Array(v).keys()).map(d => fill ?? d);
};

// math.min and math.max crashed with large arrays
export const getMax = (arr: number[]): number => {
  let len = arr.length;
  let max = -Infinity;

  while (len--) {
    max = arr[len] > max ? arr[len] : max;
  }
  return max;
};

export const getMin = (arr: number[]): number => {
  let len = arr.length;
  let min = Infinity;

  while (len--) {
    min = arr[len] < min ? arr[len] : min;
  }
  return min;
};

/**
 * @param i : range index
 * @returns : range [number, number]
 */
export const getRange = (index: number, min: number = RANGE_MIN, max: number = RANGE_MAX): [number, number] => {
  const step = (max - min) / STEP_NUM;
  return [min + index * step, min + (1 + index) * step];
};

/**
 * @param v
 * @param ranges
 * @returns whether the number v is within any of the given ranges
 */
export const withinRange = (v: number, ranges: number[][]): boolean => {
  const isInRange = (v: number, range: number[]) => v > range[0] && v <= range[1];
  const isInMin = (v: number, range: number[]) => range[0] <= RANGE_MIN && v <= range[0];
  const isInMax = (v: number, range: number[]) => range[1] >= RANGE_MAX && v > range[1];
  const flag = ranges.some(range => isInRange(v, range) || isInMin(v, range) || isInMax(v, range));

  return flag;
};

const value2rangeIdx = (v: number, min: number, max: number): number => {
  const step = (max - min) / STEP_NUM;
  return Math.floor((v - min) / step);
};

/**
 * @param samples
 * @returns historgram of each dimension
 */
export const getSampleHist = (samples: number[][]): number[][] => {
  const latentDim = samples[0].length;
  const hist = range(latentDim).map(_ => range(STEP_NUM, 0));

  // const allNums = samples.flat()
  // const rangeMin = getMin(allNums), rangeMax = getMax(allNums)

  samples.forEach(sample => {
    sample.forEach((dimensionValue, dimIdx) => {
      const idx = value2rangeIdx(dimensionValue, RANGE_MIN, RANGE_MAX);
      if (idx < 0) {
        hist[dimIdx][0] += 1;
      } else if (idx < STEP_NUM) {
        hist[dimIdx][idx] += 1;
      } else {
        hist[dimIdx][STEP_NUM - 1] += 1;
      }
    });
  });
  return hist;
};

export const generateDistribution = (
  samples: string[] | number[],
  isCategorical: boolean,
  binNum: number | undefined
): { histogram: number[]; labels: string[] } => {
  if (isCategorical) return countingCategories(samples);
  else return generateHistogram(samples as number[], binNum);
};

const countingCategories = (samples: string[] | number[]): { histogram: number[]; labels: string[] } => {
  var labels: string[] = [];
  var histogram: number[] = [];
  samples.forEach(sample => {
    const idx = labels.indexOf(sample.toString());
    if (idx == -1) {
      labels.push(sample.toString());
      histogram.push(1);
    } else {
      histogram[idx] += 1;
    }
  });
  return { histogram, labels };
};

const generateHistogram = (samples: number[], binNum: number = STEP_NUM): { histogram: number[]; labels: string[] } => {
  var labels: (string | number)[] = [];
  var histogram: number[] = range(STEP_NUM, 0);

  const minV = getMin(samples),
    maxV = getMax(samples);
  samples.forEach(sample => {
    const idx = value2rangeIdx(sample, minV, maxV);
    if (idx < 0) {
      histogram[0] += 1;
    } else if (idx < STEP_NUM) {
      histogram[idx] += 1;
    } else {
      histogram[STEP_NUM - 1] += 1;
    }
  });

  return {
    histogram,
    labels: range(STEP_NUM).map(idx =>
      [minV + (idx * (maxV - minV)) / STEP_NUM, minV + ((idx + 1) * (maxV - minV)) / STEP_NUM].join(', ')
    )
  };
};
