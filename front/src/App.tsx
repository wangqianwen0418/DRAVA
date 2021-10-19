import React from 'react';
import './App.css';

import sampleVectors from 'assets/real_samples_vector.json'
import {withinRange, getRange} from 'helpers';

import Grid from 'components/Grid';
import SampleBrowser from 'components/SampleBrowser';
import { GoslingVis } from 'components/Gosling';

const latentDim = 7
const stepNum = 11
const images = Array.from(
    Array(latentDim).keys()
  ).map(row_idx=>Array.from(
      Array(stepNum).keys()
    ).map(col_idx=>`assets/simu/${row_idx}_${col_idx}.png`))


interface State {
  filters: number[][]
}
export default class App extends React.Component <{}, State> {
  constructor(prop: {}){
    super(prop)
    this.state = {
      filters: Array.from(Array(latentDim).keys()).map(_=>Array.from(Array(stepNum).keys()))
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
        filters[row] = Array.from(Array(stepNum).keys())
        console.info(filters[row])
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
