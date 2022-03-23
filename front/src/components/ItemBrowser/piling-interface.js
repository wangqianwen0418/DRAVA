import { style } from 'd3-selection';
import createPilingJs, { createUmap } from 'piling.js';

/**
 * Promised-based image loading
 * @param   {string}  src  Remote image source, i.e., a URL
 * @return  {object}  Promise resolving to the image once its loaded
 */
export const loadImage = (src, option, isCrossOrigin = false) =>
  new Promise((resolve, reject) => {
    const image = new Image(option.imageSize, option.imageSize);
    if (isCrossOrigin) image.crossOrigin = 'Anonymous';
    image.onload = () => {
      resolve(image);
    };
    image.onerror = error => {
      console.error(`Could't load ${src}`);
      reject(error);
    };
    image.src = src;
    image.style.border = '1px solid gray';
  });

const createImageRenderer = option => sources =>
  Promise.all(
    sources.map(src => {
      const isCrossOrigin = true;
      return loadImage(src, option, isCrossOrigin);
    })
  );

export default async function create(element, pilingOptions) {
  const imageSize = 45;
  const { items, pileDragEnd, dims, getXSvgGroup, getYSvgGroup, dataset } = pilingOptions;

  const umap = createUmap();

  var spec = {
    dimensionalityReducer: umap,
    cellSize: imageSize,
    renderer: createImageRenderer({ imageSize }),
    items: items,
    itemSize: imageSize,
    showGrid: true,
    gridOpacity: 0.3
    // pileBorderColor: '#000000',
    // pileBorderSize: 1
  };

  if (dataset == 'sequence') {
    spec = {
      ...spec,
      pileItemOpacity: (item, i, pile) => 1 - i / pile.items.length, //opaciy piles for the sequence dataset
      pileSizeBadge: pile => pile.items.length > 1,
      pileItemOffset: [0, 0] //force all items overlaid
    };
  } else if (dataset == 'dsprites') {
    spec = {
      ...spec,
      pileItemOpacity: (item, i, pile) => 1 - i / pile.items.length, //opaciy piles for the dsprites dataset
      pileSizeBadge: pile => pile.items.length > 1,
      pileItemOffset: [0, 0] //force all items overlaid
      // pileItemRotation: (item, i, pile) => {
      //   const isNotLast = pile.items.length - 1 !== i;
      //   return +isNotLast * (Math.random() * 12 - 6);
      // }
    };
  } else {
    spec = {
      ...spec,
      // items in a pile is randomly rotated and offset
      pileItemOffset: (item, i, pile) => {
        const isNotLast = pile.items.length - 1 !== i;
        return [+isNotLast * (Math.random() * 12 - 6), +isNotLast * (Math.random() * 12 - 6)];
      },
      pileItemRotation: (item, i, pile) => {
        const isNotLast = pile.items.length - 1 !== i;
        return +isNotLast * (Math.random() * 12 - 6);
      }
      // pileItemOffset: (item, i, pile) => {
      //   return [0, +i * -3];
      // },
      // pileSizeBadge: pile => pile.items.length > 1
    };
  }

  const piling = createPilingJs(element, spec);

  piling.arrangeBy('data', [item => item[dims[0]], item => -1 * item[dims[1]]]);

  piling.subscribe('pileDragEnd', pileDragEnd);
  piling.subscribe('zoom', camera => {
    const svgXGroup = getXSvgGroup();
    const svgYGroup = getYSvgGroup();
    svgXGroup.attr(
      'transform',
      `translate(${camera.translation[0]}, 0) scale(${camera.scaling} 1)` // only update translate x and scale x
    );
    // to prevent the distortion of svg elements
    svgXGroup.selectAll('image').attr('transform', `scale(${1 / camera.scaling} 1)`);
    svgXGroup.selectAll('rect').attr('transform', `scale(${1 / camera.scaling} 1)`);
    svgXGroup.selectAll('text').attr('transform', `scale(${1 / camera.scaling} 1)`);

    svgYGroup.attr(
      'transform',
      `translate(${-1 * camera.translation[1]}, 0) scale(${camera.scaling} 1)` // only update translate x and scale x
    );
    // to prevent the distortion of svg elements
    svgYGroup.selectAll('image').attr('transform', `scale(${1 / camera.scaling} 1)`);
    svgYGroup.selectAll('rect').attr('transform', `scale(${1 / camera.scaling} 1)`);
    svgYGroup.selectAll('text').attr('transform', `scale(${1 / camera.scaling} 1)`);
  });

  // a set of functions to be called
  const actions = {
    reArrange: dims => {
      const [dimX, dimY] = dims;

      if (dimY == 'std') {
        const dimNum = parseInt(dimX.split('_')[1]);
        piling.arrangeBy('data', [item => item[dimX], item => -1 * item['std'][dimNum]]);
      } else {
        piling.arrangeBy('data', [item => item[dims[0]], item => -1 * item[dims[1]]]);
      }
    },
    stackX: dim => {
      piling.arrangeBy('data', [item => item['assignments'][dim], 0]);
      piling.groupBy('category', item => item['assignments'][dim] || 0);
    },
    gridGroup: dims => {
      // piling.arrangeBy('data', [item => item['assignments'][dims[0]], item => item['assignments'][dims[1]]]);
      // piling.groupBy('category', [item => item['assignments'][dims[0]], item => item['assignments'][dims[1]]]);
      piling.groupBy('grid');
    },
    splitAll: dims => {
      piling.splitAll();
      piling.arrangeBy('data', [item => item[dims[0]], item => -1 * item[dims[1]]]);
    },
    UMAP: () => {
      piling.arrangeBy('uv', 'z');
    },
    grid: dim => {
      piling.arrangeBy('data', dim);
    },
    changeSize: size => {
      piling.set({
        itemSize: size,
        renderer: createImageRenderer({ size })
      });
    },
    changeSummary: sType => {
      if (sType == 'foreshortened') {
        piling.set({
          pileItemOpacity: 1, //opaciy piles for the dsprites dataset
          pileItemOffset: (_, i, pile) => [0, i * -3] //force all items overlaid
        });
      } else if (sType == 'combining') {
        piling.set({
          pileItemOpacity: (item, i, pile) => 1 - i / pile.items.length,
          pileItemOffset: [0, 0] //force all items overlaid
        });
      } else if (sType == 'combining2') {
        // combine with offset
        piling.set({
          // pileItemOpacity: (item, i, pile) => 0.4 + (0.6 * i) / pile.items.length,
          pileItemOpacity: 0.4,
          pileItemOffset: (_, i, pile) => [0, i * -3] //force all items overlaid
        });
      } else {
        piling.set({
          pileItemOpacity: 1,
          pileItemOffset: (item, i, pile) => {
            const isNotLast = pile.items.length - 1 !== i;
            return [+isNotLast * (Math.random() * 12 - 6), +isNotLast * (Math.random() * 12 - 6)];
          },
          pileItemRotation: (item, i, pile) => {
            const isNotLast = pile.items.length - 1 !== i;
            return +isNotLast * (Math.random() * 12 - 6);
          }
        });
      }
    },
    addLabel: label => {
      piling.set({
        pileLabel: item => item[label] || '',
        pileLabelText: true,
        pileLabelTextMapping: (label, _) => (label == '0' ? 'neg' : 'pos')
      });
    }
  };

  return [piling, actions];
}
