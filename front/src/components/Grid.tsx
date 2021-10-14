import React from 'react';
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
        return <div className='grid'>
            <h4> Latent Space </h4>
            {this.props.images.map((row,row_idx) =>{
                return <div className='row' key={`row_${row_idx}`}>
                    <span className={styles.dimHeader}>DIM_{row_idx}</span>
                    {row.map((url, col_idx)=>{
                        const isSelected = filters[row_idx].includes(col_idx)
                        return <img 
                            key={`${row_idx}_${col_idx}`} 
                            src={url} 
                            className={clsx(styles.latentImage, isSelected && styles.isSelected )} 
                            alt={url} 
                            onClick = {()=>this.props.setFilters(row_idx, col_idx)}
                            />
                        })}
                </div>
            })}
            </div>
    }
}