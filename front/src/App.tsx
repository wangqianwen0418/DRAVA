import React from 'react';
import './App.css';

import Grid from './components/Grid';

const latentDim = 7
const images = Array.from(
    Array(latentDim).keys()
  ).map(row_idx=>Array.from(
      Array(11).keys()
    ).map(col_idx=>`assets/simu/${row_idx}_${col_idx}.png`))

function App() {
  return (
    <div className="App">
      <Grid images= {images}/>
    </div>
  );
}

export default App;
