import React from 'react';
import hist from 'assets/hist.json'

import styles from './Grid.module.css';
import clsx from 'clsx';

interface Props {
    images: string[][];
    filters: number[][];
    setFilters: (row:number, col:number)=> void;
}
interface States {}

export default class Grid extends React.Component <Props, States> {
    render(){
        const {filters} = this.props
        const spanWidth = 80, barHeight = 30, imgWidth = 64, barLabelHeight = 14, gap = 2
        const maxV = Math.max(...hist.flat())

        return <div className='grid'>
            <h4> Latent Space </h4>
            {this.props.images.map((row,row_idx) =>{
                return <div className={clsx(styles.rowContainer)} key={`row_${row_idx}`}>

                    <svg height={barHeight +  barLabelHeight } width={ (imgWidth+gap )*11 + spanWidth}>
                        
                        { hist[row_idx]
                            .map((h, i)=>
                                <g key={`bar_${i}`} >
                                    
                                    <rect 
                                        height={barHeight/maxV*h} 
                                        width={imgWidth} 
                                        y = {barHeight - barHeight/maxV*h }
                                        x={spanWidth + (imgWidth+gap) *i } 
                                        fill="black"
                                    />
                                    <text 
                                        x={spanWidth + (imgWidth+gap) * (i+0.5) } 
                                        y={barHeight+barLabelHeight} 
                                        textAnchor='middle'
                                    > 
                                        {h} 
                                    </text>
                                </g>
                            )
                        }
                    </svg>
                    <div>
                        <span 
                            className={styles.dimHeader} 
                            onClick={()=>this.props.setFilters(row_idx, -1)}
                            style={{width: spanWidth}}
                        >
                            DIM_{row_idx}
                        </span>
                            
                        {row.map((url, col_idx)=>{
                            const isSelected = filters[row_idx].includes(col_idx)
                            return <img 
                                key={`${row_idx}_${col_idx}`} 
                                src={url} 
                                className={clsx(styles.latentImage, isSelected && styles.isSelected )} 
                                style={{ width: imgWidth }}
                                alt={url} 
                                onClick = {()=>this.props.setFilters(row_idx, col_idx)}
                                />
                            })}
                    </div>
                </div>
            })}
            </div>
    }
}