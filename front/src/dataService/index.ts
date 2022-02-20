import axios from 'axios';
import { BASE_URL } from 'Const';
import Papa from 'papaparse';
import { TResultRow, TCSVResultRow } from 'types';
import { getAbsSum, getSum } from 'helpers';

export const whatCHR = (dataset: string) => {
  return dataset == 'sequence' ? 7 : 5;
};

export const queryResults = async (dataset: string): Promise<TResultRow[]> => {
  if (dataset == 'celeb') {
    return queryCelebResults();
  } else if (dataset == 'matrix') {
    return queryMatrixResults();
  } else {
    return querySequenceResults();
  }
};

const queryCelebResults = async () => {
  const url = '/assets/results_celeba.csv';
  const response = await axios({
    method: 'get',
    url,
    responseType: 'text'
  });
  const pcsv = Papa.parse<TCSVResultRow>(response.data, { header: true, skipEmptyLines: true });

  const samples = pcsv.data.map((row, i) => {
    const zs = row['z'].split(',').map(d => parseFloat(d));
    const dims = zs.reduce((pre, curr, idx) => {
      const dimName = `dim_${idx}`;
      return { ...pre, [dimName]: curr };
    }, {});

    return {
      ...row,
      z: row['z'].split(',').map(d => parseFloat(d)),
      id: (i + 1).toString(),
      assignments: {},
      ...dims
    };
  });
  return samples;
};

const queryMatrixResults = async () => {
  const url = '/assets/results_chr1-5_10k_onTad.csv';
  const chr = whatCHR('matrix');

  const response = await axios({
    method: 'get',
    url,
    responseType: 'text'
  });

  const pcsv = Papa.parse<TCSVResultRow>(response.data, { header: true, skipEmptyLines: true });
  const resolution = 10000;

  const samples = pcsv.data
    .filter(d => parseInt(d.chr as any) === chr)
    .map((row, i) => {
      const zs = row['z'].split(',').map(d => parseFloat(d));
      const dims = zs.reduce((pre, curr, idx) => {
        const dimName = `dim_${idx}`;
        return { ...pre, [dimName]: curr };
      }, {});

      return {
        ...row,
        chr: parseInt(row.chr as any),
        start: parseInt(row.start as any) * resolution,
        end: parseInt(row.end as any) * resolution,
        z: row['z'].split(',').map(d => parseFloat(d)),
        id: (i + 1).toString(),
        assignments: {},
        ...dims
      };
    });

  return samples;
};

const querySequenceResults = async () => {
  const url = '/assets/results_chr7_atac.csv';
  const chr = whatCHR('sequence');

  const response = await axios({
    method: 'get',
    url,
    responseType: 'text'
  });

  const pcsv = Papa.parse<TCSVResultRow>(response.data, { header: true, skipEmptyLines: true });

  const samples = pcsv.data
    .filter(d => parseInt(d.chr as any) === chr)
    .map((row, i) => {
      const zs = row['z'].split(',').map(d => parseFloat(d));
      const dims = zs.reduce((pre, curr, idx) => {
        const dimName = `dim_${idx}`;
        return { ...pre, [dimName]: curr };
      }, {});

      return {
        ...row,
        chr: parseInt(row.chr as any),
        start: parseInt(row.start as any),
        end: parseInt(row.end as any),
        z: row['z'].split(',').map(d => parseFloat(d)),
        id: i.toString(),
        assignments: {},
        ...dims
      };
    })
    .filter(
      // only samples whose latent dim have large values
      row => row['z'].some(d => Math.abs(d) > 2)
      // .reduce((a, b) => Math.abs(a) + Math.abs(b), 0) > 0.8
    );

  // only keep one samples if two samples overlap
  var newSamples: TResultRow[] = [samples[0]];
  var lastSample = samples[0];
  for (let i = 1; i < samples.length; i++) {
    const sample = samples[i];
    if (parseInt(sample.id) == parseInt(lastSample.id) + 1) {
      if (getAbsSum(sample.z) > getAbsSum(lastSample.z)) {
        newSamples.pop();
        newSamples.push(sample);
        // do not change the last sample, in case there are more than two overlapped samples
        // TO_DO: think about what is the best to handle overlapped samples
      }
    } else {
      newSamples.push(sample);
      lastSample = sample;
    }
  }

  return newSamples;
};

export const queryRawSamples = async (samples: TResultRow) => {
  const imgURLs = await axios({
    method: 'post',
    url: '',
    data: samples
  });
};

export const querySimuImages = async (dataset: string, dim: number, z?: number[]) => {
  const z_suffix = `&z=${z?.join(',')}`;
  const url = `${BASE_URL}/api/get_simu_images?dataset=${dataset}&dim=${dim}${z ? z_suffix : ''}`;
  const res = await axios({
    method: 'get',
    url
  });
  return res.data;
};
