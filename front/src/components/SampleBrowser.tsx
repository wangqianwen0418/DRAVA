import React from 'react';
import sampleVectors from 'assets/samples_vector.json'
import sampleLabels from 'assets/sample_labels.json'
import {withinRange, getRange} from 'helpers';

import styles from './SampleBrowser.module.css';

import clsx from 'clsx'

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

        return <div className={clsx(styles.sampleContainer, 'sampleBrowser' )}>
            <h4>Data Samples [{samples.length}/{sampleVectors.length}]</h4>
            {samples.map(sampleIdx=>{
                return <img 
                    src={`assets/sample_imgs/${sampleIdx}.png`} 
                    alt={`sample_${sampleIdx}`} 
                    key={sampleIdx}
                    style={{border: 'solid black 1px'}}
                    title={`${sampleLabels[sampleIdx]} \n [${sampleVectors[sampleIdx].join(', ')}]`}
                    />
            })}
        </div>
    }

}