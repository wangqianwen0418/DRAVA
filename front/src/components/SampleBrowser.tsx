import React from 'react';
import sampleVectors from 'assets/real_samples_vector.json'
import {withinRange, getRange} from 'helpers';

interface Props {
    filters: number[][];
}



export default class SampleBrowser extends React.Component <Props, {}> {
    render(){
        const {filters} = this.props;
        let samples: number[] = [] // idx of images
        sampleVectors.forEach((sampleVector, sampleIdx)=>{
            const inRange = sampleVector.every((dimensionValue, row_idx)=>{
                const ranges = filters[row_idx].map(i=>getRange(i))
                return withinRange(dimensionValue, ranges)
            })
            if (inRange) samples.push(sampleIdx);
        })
        
        return <div className='sampleBrowser'>
            <h4>Data Samples</h4>
            {samples.map(sampleIdx=>{
                return <img 
                    src={`assets/sample_imgs/${sampleIdx}.png`} 
                    alt={`sample_${sampleIdx}`} 
                    key={sampleIdx}
                    style={{border: 'solid black 1px'}}
                    title={`[${sampleVectors[sampleIdx].join(', ')}]`}
                    />
            })}
        </div>
    }

}