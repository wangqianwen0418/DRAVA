import { stepNum } from 'Const';

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
export const getRange = (i: number): [number, number] => {
    return [-3.3 + i * 0.6, -3.3 + (1 + i) * 0.6];
};

/**
 * @param v
 * @param ranges
 * @returns whether the number v is within any of the given ranges
 */
export const withinRange = (v: number, ranges: number[][]): boolean => {
    return ranges.some(range => v >= range[0] && v <= range[1]);
};

const value2rangeIdx = (v: number, min: number, max: number): number => {
    return Math.ceil((v - min) / ((max - min) / stepNum));
};

/**
 * @param sampleVectors
 * @returns historgram of each dimension
 */
export const getSampleHist = (sampleVectors: number[][]): number[][] => {
    const latentDim = sampleVectors[0].length;
    const hist = range(latentDim).map(_ => range(stepNum, 0));

    // const allNums = sampleVectors.flat()
    // const rangeMin = getMin(allNums), rangeMax = getMax(allNums)
    const rangeMin = -3,
        rangeMax = 3;

    sampleVectors.forEach(sample => {
        sample.forEach((dimensionValue, dimIdx) => {
            const idx = value2rangeIdx(dimensionValue, rangeMin, rangeMax);
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
