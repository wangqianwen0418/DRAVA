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
            {this.props.images.map((row,i) =>{
                return <div className='row' key={`row_${i}`}>
                    {row.map(url=><img key="url" src={url} className={styles.latent} alt={url} />)}
                </div>
            })}
            </div>
    }
}