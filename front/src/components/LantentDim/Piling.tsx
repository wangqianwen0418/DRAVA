import createPilingExample from './piling-interface';
import React, { useCallback } from 'react';
import styles from './Piling.module.css';

type Item = {
  src: string;
  [key: string]: any;
};
type Props = {
  items: Item[];
};
const Pilling = (props: Props) => {
  const { items } = props;
  const pilingInitHandler = useCallback(element => {
    createPilingExample(element, items);
    // const piling: any = createPilingExample(element, items);
    // return () => piling.destroy();
  }, []);

  return <div className={styles.piling_wrapper} ref={pilingInitHandler} />;
};

export default Pilling;
