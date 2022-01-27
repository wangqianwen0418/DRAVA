import { STEP_NUM, RANGE_MAX, RANGE_MIN } from 'Const';
import { TDistribution } from 'types';

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
// can use k to avoid the long tail in histogram.
// insteading of return the largest value, returen the k - th largest value
export const getMax = (arr: number[], K = 1): number => {
  let len = arr.length;
  const max = range(K).map(d => -Infinity);

  while (len--) {
    if (arr[len] > max[0]) {
      max[0] = arr[len];
      max.sort();
    }
  }
  return max[0];
};

export const getMin = (arr: number[], K = 1): number => {
  let len = arr.length;
  const min = range(K).map(d => Infinity);

  while (len--) {
    if (arr[len] < min[K - 1]) {
      min[K - 1] = arr[len];
      min.sort();
    }
  }
  return min[K - 1];
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

export const generateDistribution = (
  samples: string[] | number[],
  isCategorical: boolean,
  binNum?: number,
  sampleIds?: string[],
  K?: number,
  valueRange?: number[] // users can specify a range to draw the histogram
): TDistribution => {
  // no meaningful data
  if (samples.length == 0) return { histogram: [], labels: [], groupedSamples: [] };
  if (samples[0] == undefined) return { histogram: [], labels: [], groupedSamples: [] };

  if (!sampleIds) {
    sampleIds = samples.map((d, idx) => idx.toString());
  }

  // meaningful data
  if (isCategorical) return countingCategories(samples, sampleIds);
  else return generateHistogram(samples as number[], binNum, sampleIds, K, valueRange);
};

const countingCategories = (samples: string[] | number[], sampleIds: string[]): TDistribution => {
  var labels: string[] = [];
  var histogram: number[] = [];
  var groupedSamples: string[][] = [];
  samples.forEach((sample, sampleIdx) => {
    const idx = labels.indexOf(sample.toString());
    if (idx == -1) {
      labels.push(sample.toString());
      histogram.push(1);
      groupedSamples.push([sampleIds[sampleIdx]]);
    } else {
      histogram[idx] += 1;
      groupedSamples[idx].push(sampleIds[sampleIdx]);
    }
  });
  return { histogram, labels, groupedSamples };
};

const generateHistogram = (
  samples: number[],
  binNum: number = STEP_NUM,
  sampleIds: string[],
  K: number = 1,
  valueRange?: number[]
): TDistribution => {
  var histogram: number[] = range(STEP_NUM, 0);
  var groupedSamples: string[][] = range(STEP_NUM).map(_ => []);

  // in case the csv parser process number to string
  samples = samples.map(d => parseFloat(d as any));

  var [minV, maxV] = valueRange || [0, 0];
  if (!valueRange) {
    minV = getMin(samples, K);
    maxV = getMax(samples, K);
  }

  samples.forEach((sample, sampleIdx) => {
    var idx = value2rangeIdx(sample, minV, maxV);
    if (idx < 0) {
      idx = 0;
    } else if (idx >= STEP_NUM) {
      idx = STEP_NUM - 1;
    }
    histogram[idx] += 1;
    groupedSamples[idx].push(sampleIds[sampleIdx]);
  });

  // to short num
  const shortNum = (num: number): string => {
    return num > 1000 ? `${Math.floor(num / 1000).toString()}k` : num.toFixed(3);
  };

  return {
    histogram,
    groupedSamples,
    labels: range(STEP_NUM).map(idx =>
      [minV + (idx * (maxV - minV)) / STEP_NUM, minV + ((idx + 1) * (maxV - minV)) / STEP_NUM]
        .map(d => shortNum(d))
        .join('~')
    )
  };
};
