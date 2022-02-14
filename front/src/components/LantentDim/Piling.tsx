import createPilingExample from './piling-interface';
import React, { useCallback, useState } from 'react';
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

  const pilingInitHandler = useCallback(
    element => {
      if (element !== null) {
        const piling: any = createPilingExample(element, items);
        return () => piling.destroy();
      }
    },
    [props.items]
  );

  return (
    <div className={styles.piling_container}>
      <div className={styles.piling_wrapper} ref={pilingInitHandler} />
    </div>
  );
};

export default Pilling;
