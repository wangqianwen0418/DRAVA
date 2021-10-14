import React from 'react';
import sampleVectors from 'assets/real_samples_vector.json'

interface Props {
    filters: number[][];
}

const idx2range = (idx:number):[number, number]=>{
    return [-3.3 + idx*0.6, -3.3 + (1+ idx)*0.6]
}

const withinRange = (v: number, ranges:number[][]):boolean =>{
    if (ranges.length === 0) return true

    ranges.forEach(range=>{
        if (v >= range[0] && v <= range[1]) return true
    })
    return false
}

export default class SampleBrowser extends React.Component <Props, {}> {
    render(){
        const {filters} = this.props;
        let samples: number[] = [] // idx of images
        sampleVectors.forEach((sampleVector, sampleIdx)=>{
            const inRange = sampleVector.every((dimensionValue, row_idx)=>{
                const ranges = filters[row_idx].map(i=>idx2range(i))
                return withinRange(dimensionValue, ranges)
            })
            if (inRange) samples.push(sampleIdx);
        })
        console.info(samples)
        return <div className='sampleBrowser'>
            <h4>Data Samples</h4>
            {samples.map(sampleIdx=>{
                return <img src={`assets/sample_imgs/${sampleIdx}.png`} alt={`sample_${sampleIdx}`} key={sampleIdx}/>
            })}
        </div>
    }

}