import React from 'react';
import './App.css';

import {latentDim, stepNum} from 'Const';
import {range} from 'helpers';
import sampleVectors from 'assets/samples_vector.json'
import {withinRange, getRange} from 'helpers';

import Grid from 'components/Grid';
import SampleBrowser from 'components/SampleBrowser';
import { GoslingVis } from 'components/Gosling';


const images = range(latentDim)
  .map(
    row_idx=>range(stepNum)
      .map(col_idx=>`assets/simu/${row_idx}_${col_idx}.png`)
    )


interface State {
  filters: number[][]
}
export default class App extends React.Component <{}, State> {
  constructor(prop: {}){
    super(prop)
    this.state = {
      filters: range(latentDim).map(_=>range(stepNum))
    }
    this.setFilters = this.setFilters.bind(this)
  }

  setFilters(row:number, col: number){
    let {filters} = this.state 

    if (col === -1){
      // set filters for the whole row
      if (filters[row].length>0) {
        filters[row] = []
      } else {
        filters[row] = range(stepNum)
      }
    } else {
      // set filters for single grids
      const idx = filters[row].indexOf(col)
      if (idx === -1) {
        filters[row].push(col)
      } else {
        filters[row].splice(idx, 1)
      }
    }
    this.setState({filters})
  }
  filterSamples (){
    const {filters} = this.state

    let sampleIdxs: number[] = [] // idx of images
        sampleVectors.forEach((sampleVector, sampleIdx)=>{
            const inRange = sampleVector.every((dimensionValue, row_idx)=>{
                const ranges = filters[row_idx].map(i=>getRange(i))
                return withinRange(dimensionValue, ranges)
            })
            if (inRange) sampleIdxs.push(sampleIdx);
        })
    return sampleIdxs
  }
  render(){
    const {filters} = this.state

    const sampleIdxs = this.filterSamples()

    return (
      <div className="App">
        <Grid images= {images} setFilters = {this.setFilters} filters={filters}/>
        <GoslingVis sampleIdxs={sampleIdxs}/>
        <SampleBrowser sampleIdxs={sampleIdxs}/>
      </div>
    );
  }
}
