import React from 'react';
import './App.css';
import { Row, Col } from 'antd';

import {stepNum} from 'Const';
import {range} from 'helpers';
import {withinRange, getRange} from 'helpers';

import Grid from 'components/Grid';
import SampleBrowser from 'components/SampleBrowser';
import { GoslingVis } from 'components/Gosling';

import {requestHist} from 'dataService';


interface State {
  filters: number[][],
  hist: number[][]
}
export default class App extends React.Component <{}, State> {
  constructor(prop: {}){
    super(prop)
    this.state = {
      filters: [],
      hist: []
    }
    this.setFilters = this.setFilters.bind(this)
  }

  async onRequestHist() {
    const hist = await requestHist() 
    const filters = range(hist.length).map(_=>range(stepNum))
    this.setState({filters, hist})
  }
 componentDidMount(){
     this.onRequestHist()
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
  // filterSamples (){
  //   const {filters} = this.state

  //   let sampleIdxs: number[] = [] // idx of images
  //       sampleVectors.forEach((sampleVector, sampleIdx)=>{
  //           const inRange = sampleVector.every((dimensionValue, row_idx)=>{
  //               const ranges = filters[row_idx].map(i=>getRange(i))
  //               return withinRange(dimensionValue, ranges)
  //           })
  //           if (inRange) sampleIdxs.push(sampleIdx);
  //       })
  //   return sampleIdxs
  // }
  render(){
    const {filters, hist} = this.state
    if (filters.length === 0) return null 

    // const sampleIdxs = this.filterSamples()
    const sampleIdxs = range(400)
    const appPadding = 10, gutter = 16, appHeight = window.innerHeight - appPadding * 2, 
      colWidth = (window.innerWidth - appPadding * 2)*0.5 - gutter

    return (
      <div className="App" style={{padding: appPadding}}>
      <Row gutter={gutter}>

        <Col span={12}>
          <GoslingVis sampleIdxs={sampleIdxs} width={colWidth} height={appHeight * 0.5} />
          <SampleBrowser sampleIdxs={sampleIdxs} height={appHeight * 0.5}/>
        </Col>

        <Col span={12}>
          <Grid setFilters = {this.setFilters} filters={filters} height={ appHeight } width={colWidth} hist={hist}/>
        </Col>
      </Row>
      </div>
    );
  }
}
