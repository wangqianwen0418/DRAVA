import axios from 'axios';
import { BASE_URL } from 'Const';
import Papa from 'papaparse';
import { TResultRow, TCSVResultRow } from 'types';
import { getAbsSum, getSum } from 'helpers';

export const whatCHR = (dataset: string) => {
  return dataset == 'sequence' ? 7 : 5;
};

export const queryResults = async (dataset: string): Promise<TResultRow[]> => {
  try {
    const cap_name = `query${dataset[0].toUpperCase() + dataset.substring(1)}Results`;
    return queryFunctions[cap_name]();
  } catch {
    return defaultResultsQuery(dataset);
  }
};

/***
 * Custom query functions
 */
const defaultResultsQuery = async (dataset: string) => {
  const url = `/assets/results_${dataset}.csv`;
  try {
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
        z: row['z'].split(',').map(d => parseFloat(d) / 6 + 0.5),
        std: row['std'].split(',').map(d => parseFloat(d)),
        id: row['id'],
        assignments: {},
        ...dims,
        index: i
      };
    });
    return samples;
  } catch (err) {
    console.error(err);
    return [];
  }
};

const queryIDCResults = async () => {
  // const url = '/assets/results_IDC_10285.csv';
  const url = '/assets/results_IDC.csv';
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
      z: row['z'].split(',').map(d => parseFloat(d) / 6 + 0.5),
      std: row['std'].split(',').map(d => parseFloat(d)),
      id: row['img_path'],
      assignments: {},
      ...dims,
      prediction: row['acc'] == 'True' ? 'right' : 'wrong',
      index: i
    };
  });

  return samples;
};

const queryDspritesResults = async () => {
  const url = '/assets/results_dsprites.csv';
  const response = await axios({
    method: 'get',
    url,
    responseType: 'text'
  });
  const pcsv = Papa.parse<TCSVResultRow>(response.data, { header: true, skipEmptyLines: true });

  const samples = pcsv.data.map((row, i) => {
    const zs = row['z'].split(',').map((d, i) => (i == 4 ? -1 : 1) * parseFloat(d));
    const dims = zs.reduce((pre, curr, idx) => {
      const dimName = `dim_${idx}`;
      return { ...pre, [dimName]: curr };
    }, {});

    return {
      ...row,
      embedding: [2, 3, 4, 7, 0].map(i => zs[i] / 3 + 0.5),
      z: row['z'].split(',').map(d => parseFloat(d)),
      std: row['std'].split(',').map(d => parseFloat(d)),
      id: i.toString(),
      assignments: {},
      ...dims,
      index: i
    };
  });
  return samples;
};

const queryCelebaResults = async () => {
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
      std: row['std'].split(',').map(d => parseFloat(d)),
      id: (i + 1).toString(),
      assignments: {},
      ...dims,
      index: i
    };
  });
  return samples;
};

const querySc2Results = async () => {
  const url = '/assets/results_sc2_labeled.csv';
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
      ...dims,
      index: i
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
  const resolution = 10000; //10k

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
        std: row['std'].split(',').map(d => parseFloat(d)),
        id: (i + 1).toString(),
        index: i,
        size: parseInt((row.end - row.start) as any) * resolution,
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
        std: row['std'].split(',').map(d => parseFloat(d)),
        id: i.toString(),
        assignments: {},
        ...dims
      };
    })
    .filter(
      // only samples whose latent dim have large values
      // TO-DO: need a baseline to know which metric is better
      // row => row['z'].some(d => Math.abs(d) > 1.5)
      row => getAbsSum(row['z']) > 5
    );

  // only keep one sample with larger latent values if two samples overlap
  var newSamples: Partial<TResultRow>[] = [samples[0]];
  var lastSample = samples[0];
  for (let i = 1; i < samples.length; i++) {
    const sample = samples[i];
    if (parseInt(sample.id) == parseInt(lastSample.id) + 1) {
      if (getAbsSum(sample.z) > getAbsSum(lastSample.z)) {
        newSamples.pop();
        newSamples.push(sample);
        lastSample = sample;
        // TO_DO: think about what is the best to handle overlapped samples
      }
    } else {
      newSamples.push(sample);
      lastSample = sample;
    }
  }
  newSamples = newSamples.map((d, i) => ({ ...d, index: i }));

  return newSamples as TResultRow[];
};

const queryFunctions: { [k: string]: () => Promise<TResultRow[]> } = {
  queryCelebaResults,
  queryDspritesResults,
  queryIDCResults,
  queryMatrixResults,
  querySc2Results,
  querySequenceResults
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
