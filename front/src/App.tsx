import React from 'react';
import './App.css';

import {latentDim, stepNum} from 'Const';
import {range} from 'helpers';

import Grid from 'components/Grid';
import SampleBrowser from 'components/SampleBrowser';


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
  render(){
    const {filters} = this.state
    return (
      <div className="App">
        <Grid images= {images} setFilters = {this.setFilters} filters={filters}/>
        <SampleBrowser filters={filters}/>
      </div>
    );
  }
}
