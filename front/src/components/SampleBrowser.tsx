import React from 'react';
import sampleVectors from 'assets/samples_vector.json'
import sampleLabels from 'assets/sample_labels.json'

import { Card } from 'antd';

import styles from './SampleBrowser.module.css';

import clsx from 'clsx'

interface Props {
    sampleIdxs: number[],
    height: number
}



export default class SampleBrowser extends React.Component <Props, {}> {
    render(){
        const {sampleIdxs, height} = this.props;
        
        const rootStyle = getComputedStyle(document.documentElement),
        cardHeadHeight = parseInt( rootStyle.getPropertyValue('--card-head-height') )
        return <Card title="Samples"  size="small" bodyStyle={{overflowY:'scroll', height: height - cardHeadHeight}}>
            {sampleIdxs.map(sampleIdx=>{
                return <img 
                    src={`assets/sample_imgs/${sampleIdx}.png`} 
                    alt={`sample_${sampleIdx}`} 
                    key={sampleIdx}
                    style={{border: 'solid black 1px'}}
                    title={`${sampleLabels[sampleIdx]} \n [${sampleVectors[sampleIdx].join(', ')}]`}
                    />
            })}
        </Card>
    }

}