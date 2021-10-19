import React from 'react';
import sampleVectors from 'assets/real_samples_vector.json'

import styles from './SampleBrowser.module.css';

import clsx from 'clsx'

interface Props {
    sampleIdxs: number[]
}



export default class SampleBrowser extends React.Component <Props, {}> {
    render(){
        const {sampleIdxs} = this.props;
        

        return <div className={clsx(styles.sampleContainer, 'sampleBrowser' )}>
            <h4>Data Samples</h4>
            {sampleIdxs.map(sampleIdx=>{
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