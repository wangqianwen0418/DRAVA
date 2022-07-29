import axios from 'axios';
import { BASE_URL } from 'Const';
import { TResultRow } from 'types';

export const whatCHR = (dataset: string) => {
  return dataset == 'sequence' ? 7 : 5;
};

export const queryResults = async (dataset: string): Promise<TResultRow[]> => {
  const url = `${BASE_URL}/api/get_model_results?dataset=${dataset}`;
  const res = await axios({
    method: 'get',
    url
  });
  return res.data;
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

export const queryItem = async (dataset: string, id: string) => {
  const url = `${BASE_URL}/api/get_item_sample?dataset=${dataset}&id=${id}`;
  const res = await axios({
    method: 'get',
    url
  });
  return res.data;
};
