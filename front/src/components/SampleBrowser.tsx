import React from 'react';
import sampleVectors from 'assets/samples_vector.json'
import sampleLabels from 'assets/sample_labels.json'

import styles from './SampleBrowser.module.css';

import clsx from 'clsx'

interface Props {
    sampleIdxs: number[]
}



export default class SampleBrowser extends React.Component <Props, {}> {
    render(){
        const {sampleIdxs} = this.props;
        

        return <div className={clsx(styles.sampleContainer, 'sampleBrowser' )}>
            <h4>Data Samples [{sampleIdxs.length}/{sampleVectors.length}]</h4>
            {sampleIdxs.map(sampleIdx=>{
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