import axios from 'axios';
import Papa from 'papaparse';
import { TResultRow, TCSVResultRow } from 'types';

export const whatCHR = (dataset: string) => {
  return dataset == 'sequence' ? 7 : 5;
};

export const queryResults = async (dataset: string): Promise<TResultRow[]> => {
  if (dataset == 'celeb') {
    const url = '/assets/results_celeba.csv';
    const response = await axios({
      method: 'get',
      url,
      responseType: 'text'
    });
    const pcsv = Papa.parse<TCSVResultRow>(response.data, { header: true, skipEmptyLines: true });

    const samples = pcsv.data.map((row, i) => {
      return {
        ...row,
        z: row['z'].split(',').map(d => parseFloat(d)),
        id: (i + 1).toString()
      };
    });
    return samples;
  }

  const url = dataset == 'sequence' ? '/assets/results_chr7_atac.csv' : '/assets/results_chr1-5_10k_onTad.csv';
  const chr = whatCHR(dataset);

  const response = await axios({
    method: 'get',
    url,
    responseType: 'text'
  });

  const pcsv = Papa.parse<TCSVResultRow>(response.data, { header: true, skipEmptyLines: true });
  const resolution = dataset == 'sequence' ? 1 : 10000;

  const samples = pcsv.data
    .filter(d => parseInt(d.chr as any) === chr)
    .map((row, i) => {
      return {
        ...row,
        chr: parseInt(row.chr as any),
        start: parseInt(row.start as any) * resolution,
        end: parseInt(row.end as any) * resolution,
        z: row['z'].split(',').map(d => parseFloat(d)),
        id: (dataset == 'sequence' ? i : i + 1).toString()
      };
    })
    .filter(
      // only samples whose latent dim have large values
      row => dataset !== 'sequence' || row['z'].some(d => Math.abs(d) > 1.5)
      // .reduce((a, b) => Math.abs(a) + Math.abs(b), 0) > 0.8
    );

  return samples;
};

export const queryRawSamples = async (samples: TResultRow) => {
  const imgURLs = await axios({
    method: 'post',
    url: '',
    data: samples
  });
};
