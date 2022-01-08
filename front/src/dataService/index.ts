import axios from 'axios';
import { stratify } from 'd3-hierarchy';
import { getSampleHist } from 'helpers';
import Papa from 'papaparse';
import { TResultRow, TCSVResultRow } from 'types';

export const queryResults = async (dataset: string): Promise<TResultRow[]> => {
  const url = dataset == 'sequence' ? '/assets/results_chr7_atac.csv' : '/assets/test:results_chr1-5_10k_onTad.csv';

  const response = await axios({
    method: 'get',
    url,
    responseType: 'text'
  });

  const pcsv = Papa.parse<TCSVResultRow>(response.data, { header: true, skipEmptyLines: true });
  const resolution = dataset == 'sequence' ? 1 : 10000;

  const samples = pcsv.data.map((row, i) => {
    return {
      ...row,
      chr: parseInt(row.chr as any),
      start: parseInt(row.start as any) * resolution,
      end: parseInt(row.end as any) * resolution,
      z: row['z'].split(',').map(d => parseFloat(d)),
      id: i.toString()
    };
  });

  return samples;
};
