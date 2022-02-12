import createPilingExample from './piling-interface';
import React, { useCallback, useState } from 'react';
import styles from './Piling.module.css';

import { Select } from 'antd';
import { TResultRow } from 'types';
import { BASE_URL } from 'Const';

const { Option } = Select;

type Item = TResultRow & {
  src: string;
  [key: string]: any;
};
type Props = {
  samples: TResultRow[];
  dataset: string;
  dimX: string;
};
const Pilling = (props: Props) => {
  const { samples, dataset, dimX } = props;

  const [dimY, changeDimY] = useState('dim_7');
  const dimXNum = parseInt(dimX.split('_')[1]);
  const dimYNum = parseInt(dimY.split('_')[1]);

  const items = samples.map(s => {
    const url = `${BASE_URL}/api/get_${dataset}_sample?id=${s.id}`;
    return { ...s, src: url, x: s.z[dimXNum], y: s.z[dimYNum] };
  });

  const pilingInitHandler = useCallback(
    element => {
      if (element !== null) {
        const piling: any = createPilingExample(element, items);
        return () => piling.destroy();
      }
    },
    [dimY]
  );

  const ySelector = (
    <Select onChange={(e: string) => changeDimY(e)} value={dimY}>
      <Option value="dim_7">dim_7</Option>
      <Option value="dim_0">dim_0</Option>
    </Select>
  );

  return (
    <div className={styles.piling_container}>
      <h3> All samples are horizontally oragnized by {dimX} </h3>
      <label>Oragnize samples vertically using </label> {ySelector}
      <div className={styles.piling_wrapper} ref={pilingInitHandler} />
    </div>
  );
};

export default Pilling;
