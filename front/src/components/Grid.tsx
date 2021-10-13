import React from 'react';
import styles from './Grid.module.css';

interface Props {
    images: string[][];
}
interface States {}

export default class Grid extends React.Component <Props, States> {
    render(){
        return <div className='grid'>
            <h4> Latent Space </h4>
            {this.props.images.map((row,row_idx) =>{
                return <div className='row' key={`row_${row_idx}`}>
                    <span className={styles.dimHeader}>DIM_{row_idx}</span>
                    {row.map((url, col_idx)=><img key={`${row_idx}_${col_idx}`} src={url} className={styles.latentImage} alt={url} />)}
                </div>
            })}
            </div>
    }
}