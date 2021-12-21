import axios from 'axios';
import { stratify } from 'd3-hierarchy';
import { getSampleHist } from 'helpers';
import Papa from 'papaparse';
import { TResultRow, TCSVResultRow } from 'types';

export const queryResults = async (dataset: string): Promise<TResultRow[]> => {
    const url = '/assets/results_chr7_atac.csv';
    if (dataset == 'sequence') {
        // do some thing
    }
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
            id: i.toString()
        };
    });

    return samples;
};
