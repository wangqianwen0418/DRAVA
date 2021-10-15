export const getRange = (i:number):[number, number]=>{
    return [-3.3 + i * 0.6, -3.3 + (1+ i)*0.6]
}

export const withinRange = (v: number, ranges:number[][]):boolean =>{
    return ranges.some(range=> (v >= range[0] && v <= range[1]))
}

