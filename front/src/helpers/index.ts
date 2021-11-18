import { latentDim, stepNum } from "Const"

export const getRange = (i:number):[number, number]=>{
    return [-3.3 + i * 0.6, -3.3 + (1+ i)*0.6]
}

/**
 * 
 * @param v : length of array
 * @param fill: 
 * - when given, fill the array with this constant value 
 * - otherwise, [0, 1, .., v-1]
 * @returns the array
 */
export const range = (v : number, fill?: number):number[] => {
    return Array.from(Array(v).keys()).map(d=>fill ?? d)
}

/**
 * @param v 
 * @param ranges 
 * @returns whether the number v is within any of the ranges
 */
export const withinRange = (v: number, ranges:number[][]):boolean =>{
    return ranges.some(range=> (v >= range[0] && v <= range[1]))
}

const value2rangeIdx = (v:number):number => {
    return Math.floor( (v + 3.3 )/0.6 )
}



/**
 * @param sampleVectors 
 * @returns historgram of each dimension 
 */
export const getSampleHist = (sampleVectors: number[][]): number[][]=>{
    let hist = range(latentDim).map( _ => range(stepNum, 0))
    sampleVectors.forEach(sample=>{
        sample.forEach((dimensionValue, dimIdx)=>{
            const idx = value2rangeIdx(dimensionValue)
            if (idx <  stepNum) {
                hist[dimIdx][idx] += 1
            }
        })
    })
    return hist
}

