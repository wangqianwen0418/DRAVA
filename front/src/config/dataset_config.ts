type viewName = 'latentDim' | 'gosling' | 'itemView' | 'contextView';
type TDatasetConfig = {
  [key: string]: {
    name: string;
    labels: string[];
    customDims: string[];
    views: { left: viewName[]; right: viewName[] };
    imgSize?: number;
  };
};

const datasetConfig: TDatasetConfig = {
  matrix: {
    name: 'Matrix',
    labels: [],
    customDims: [
      'size',
      'score',
      'ctcf_mean',
      'ctcf_left',
      'ctcf_right',
      'atac_mean',
      'atac_left',
      'atac_right',
      'recons_loss'
    ],
    views: { left: ['latentDim', 'gosling'], right: ['itemView'] }
  },
  celeba: {
    name: 'CelebA',
    labels: ['gender', 'smiling', 'hair', 'bangs', 'young'],
    customDims: ['recons_loss'],
    views: { left: ['latentDim'], right: ['itemView'] }
  },
  // sequence: {
  //   name: 'Genomic Sequence',
  //   labels: [],
  //   customDims: ['peak_score', 'recons_loss'],
  //   views: { left: ['latentDim', 'gosling'], right: ['itemView'] }
  // },
  IDC: {
    name: 'Breaset Cancer',
    labels: ['label', 'prediction'],
    customDims: ['label', 'confidence', 'prediction', 'recons_loss'],
    views: { left: ['latentDim', 'contextView'], right: ['itemView'] },
    imgSize: 50
  },
  dsprites: {
    name: 'Shapes',
    labels: [],
    customDims: ['recons_loss'],
    views: { left: ['latentDim'], right: ['itemView'] }
  },
  sc2: {
    name: 'Single Cell',
    labels: [
      'K-Means [Mean] Expression',
      'K-Means [Covariance] Expression',
      'K-Means [Total] Expression',
      'K-Means [Mean-All-SubRegions] Expression',
      'K-Means [Shape-Vectors]',
      'K-Means [Texture]',
      'K-Means [tSNE_All_Features]'
    ],
    customDims: ['K-Means [Total] Expression'],
    views: { left: ['latentDim', 'contextView'], right: ['itemView'] },
    imgSize: 64
  }
};

export { datasetConfig };
