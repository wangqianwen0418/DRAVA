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
export const getSampleHist = (samples: number[][]): TDistribution[] => {
  const latentDim = samples[0].length;
  var results = range(latentDim).map(i => {
    const sampleValues = samples.map(sample => sample[i]);
    return generateDistribution(sampleValues, false, STEP_NUM, [RANGE_MIN, RANGE_MAX]);
  });
  return results;
};

export const generateDistribution = (
  samples: string[] | number[],
  isCategorical: boolean,
  binNum?: number,
  valueRange?: number[], // users can specify a range to draw the histogram
  sampleIds?: string[]
): TDistribution => {
  // no meaningful data
  if (samples.length == 0) return { histogram: [], labels: [], groupedSamples: [] };
  if (samples[0] == undefined) return { histogram: [], labels: [], groupedSamples: [] };

  if (!sampleIds) {
    sampleIds = samples.map((d, idx) => idx.toString());
  }

  // meaningful data
  if (isCategorical) return countingCategories(samples, sampleIds);
  else return generateHistogram(samples as number[], binNum, sampleIds, valueRange);
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
  valueRange?: number[]
): TDistribution => {
  var histogram: number[] = range(STEP_NUM, 0);
  var groupedSamples: string[][] = range(STEP_NUM).map(_ => []);

  // in case the csv parser process number to string
  samples = samples.map(d => parseFloat(d as any));

  var [minV, maxV] = valueRange || [0, 0];
  if (!valueRange) {
    minV = getMin(samples);
    maxV = getMax(samples);
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
