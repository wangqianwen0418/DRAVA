import { stepNum } from 'Const';
const RANGE_MIN = -3;
const RANGE_MAX = 3;

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
    const step = (max - min) / stepNum;
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
    const step = (max - min) / stepNum;
    return Math.floor((v - min) / step);
};

/**
 * @param samples
 * @returns historgram of each dimension
 */
export const getSampleHist = (samples: number[][]): number[][] => {
    const latentDim = samples[0].length;
    const hist = range(latentDim).map(_ => range(stepNum, 0));

    // const allNums = samples.flat()
    // const rangeMin = getMin(allNums), rangeMax = getMax(allNums)

    samples.forEach(sample => {
        sample.forEach((dimensionValue, dimIdx) => {
            const idx = value2rangeIdx(dimensionValue, RANGE_MIN, RANGE_MAX);
            if (idx < 0) {
                hist[dimIdx][0] += 1;
            } else if (idx < stepNum) {
                hist[dimIdx][idx] += 1;
            } else {
                hist[dimIdx][stepNum - 1] += 1;
            }
        });
    });
    return hist;
};
