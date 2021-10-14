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
        const maxV = Math.max(...hist.flat()), barHeight = 40, imgWidth = 64, barWidth = 10

        return <div className='grid'>
            <h4> Latent Space </h4>
            {this.props.images.map((row,row_idx) =>{
                return <div className='row' key={`row_${row_idx}`}>

                    <svg height={40} width={imgWidth*11}>{ hist[row_idx]
                            .map((h, i)=>
                                <rect key={`bar_${i}`} 
                                    height={barHeight/maxV*h} 
                                    width={barWidth} 
                                    x={imgWidth*i + (imgWidth - barWidth)/2} 
                                    fill="black"
                                />
                                )
                        }
                    </svg>

                    <span 
                        className={styles.dimHeader} 
                        onClick={()=>this.props.setFilters(row_idx, -1)}
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
            })}
            </div>
    }
}