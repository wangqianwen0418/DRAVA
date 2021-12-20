import axios from 'axios';
import { getSampleHist } from 'helpers';
import Papa from 'papaparse';

type TResultRow = {
    chr: number;
    start: number;
    end: number;
    z: string; // '[x, x, x,]',
    [key: string]: any;
};

export const requestHist = async () => {
    const url = '/assets/results_chr7_atac.csv';
    const response = await axios({
        method: 'get',
        url,
        responseType: 'text'
    });

    const pcsv = Papa.parse<TResultRow>(response.data, { header: true, skipEmptyLines: true });
    const z = pcsv.data.map(row => row['z'].split(',').map(d => parseFloat(d)));
    const hist = getSampleHist(z);
    return hist;
};
