import createPilingExample from './piling-interface';
import React, { useCallback, useEffect, useState } from 'react';
import styles from './Piling.module.css';

import { TResultRow } from 'types';

type Item = TResultRow & {
  src: string;
  [key: string]: any;
};
type Props = {
  items: Item[];
};
const Pilling = (props: Props) => {
  const { items } = props;
  const pileDragEnd = (e: any) => console.info('end of piling drag, ', e.target.items);

  const pilingInitHandler = useCallback(element => {
    if (element !== null) {
      createPilingExample(element, items, pileDragEnd);
    }
    return;
  }, []);

  return (
    <div className={styles.piling_container}>
      <div className={styles.piling_wrapper} ref={pilingInitHandler} />
    </div>
  );
};

export default Pilling;
