import { postNewGroups } from 'dataService';
import { message } from 'antd';
import createPilingJs, { createUmap } from 'piling.js';

import { IS_ONLINE_DEMO } from 'Const';

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
  const { items, dims, getXSvgGroup, getYSvgGroup, dataset } = pilingOptions;

  const umap = createUmap();

  var spec = {
    dimensionalityReducer: umap,
    cellSize: imageSize,
    cellPadding: 3,
    renderer: createImageRenderer({ imageSize }),
    items: items,
    itemSize: imageSize,
    showGrid: true,
    gridOpacity: 0.3,
    pileItemRotation: 0,
    pileSizeBadge: pile => pile.items.length > 1,
    pileLabelSizeTransform: 'histogram',
    depileMethod: 'hoveredOne',
    pileOrderItems: pileState =>
      pileState.items.sort((a, b) => items[+a - 1]['recons_loss'] || 0 - items[+b - 1]['recons_loss'] || 0)
    // pileLabelStackAlign: 'vertical'
    // pileBorderColor: '#000000',
    // pileBorderSize: 1
  };

  if (dataset == 'sequence') {
    spec = {
      ...spec,
      pileItemOpacity: (item, i, pile) => 1 - i / pile.items.length, //opaciy piles for the sequence dataset
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

  // UMAP project by default
  piling.arrangeBy('uv', 'embedding');

  // piling.subscribe('zoom', camera => {
  //   const svgXGroup = getXSvgGroup();
  //   const svgYGroup = getYSvgGroup();
  //   svgXGroup.attr(
  //     'transform',
  //     `translate(${camera.translation[0]}, 0) scale(${camera.scaling} 1)` // only update translate x and scale x
  //   );
  //   // to prevent the distortion of svg elements
  //   svgXGroup.selectAll('image').attr('transform', `scale(${1 / camera.scaling} 1)`);
  //   svgXGroup.selectAll('rect').attr('transform', `scale(${1 / camera.scaling} 1)`);
  //   // svgXGroup.selectAll('text').attr('transform', `scale(${1 / camera.scaling} 1)`);

  //   svgYGroup.attr(
  //     'transform',
  //     `translate(${-1 * camera.translation[1]}, 0) scale(${camera.scaling} 1)` // x, y are switched since yGroup is rotated by 90deg
  //   );
  //   // to prevent the distortion of svg elements
  //   svgYGroup.selectAll('image').attr('transform', `translate(0, ${camera.translation[1]}) scale(1 ${1 / camera.scaling})`);
  //   svgYGroup.selectAll('rect').attr('transform', `translate(0, ${camera.translation[1]}) scale(1 ${1 / camera.scaling})`);
  //   // svgYGroup.selectAll('text').attr('transform', `scale(${1 / camera.scaling} 1)`);
  // });

  // a set of functions to be called
  const actions = {
    postNewGroups: () => {
      if (IS_ONLINE_DEMO) {
        message.warning(
          'Update Concept is not supported in the online demo.\n Please download Drava and run it on your local computer.',
          5 //duration = 5s
        );
      } else {
        const currentPiles = Object.values(piling.exportState()['piles'])
          .filter(d => d.items.length > 0)
          .sort((a, b) => a.x - b.x);

        postNewGroups(currentPiles);
      }
    },
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
      piling.groupBy('category', item => item['assignments'][dim] || 0);
    },
    gridGroup: dims => {
      // piling.groupBy('category', [
      //   item => Math.floor(item['assignments'][dims[0]] / 2),
      //   item => -1 * Math.floor(item['assignments'][dims[1]] / 2)
      // ]);
      piling.groupBy('category', [item => item['assignments'][dims[0]], item => -1 * item['assignments'][dims[1]]]);
      // piling.groupBy('grid');
      // piling.arrangeBy('data', [item => item['assignments'][dims[0]], item => -1 * item['assignments'][dims[1]]]);
      // piling.groupBy('grid', { columns: 21, cellAspectRatio: 1 });
      piling.set({
        pileItemRotation: 0,
        pileItemOffset: [0, 0]
      });
      // piling.groupBy('category', [item => item['assignments'][dims[0]], item => -1 * item['assignments'][dims[1]]]);
    },
    splitAll: dims => {
      piling.splitAll();
      // piling.arrangeBy('data', [item => item[dims[0]], item => -1 * item[dims[1]]]);
    },
    UMAP: () => {
      if (dataset == 'dsprites') {
        piling.arrangeBy('uv', 'embedding');
      } else {
        piling.arrangeBy('uv', 'z');
      }
    },
    grid: dim => {
      piling.arrangeBy('data', dim);
    },
    grid2D: dims => {
      piling.arrangeBy('data', [item => item['assignments'][dims[0]], item => -1 * item['assignments'][dims[1]]]);
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
          pileItemRotation: 0,
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
          pileItemOpacity: (item, i, pile) => (pile.items.length > 1 ? 0.6 : 1),
          pileItemOffset: (_, i, pile) => [0, i * -5] //force all items overlaid
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
      if (dataset == 'IDC') {
        piling.set({
          pileLabel: item => item[label] || '',
          pileLabelText: true,
          pileLabelColor: ['#3295a8', '#e0722b'],
          pileLabelFontSize: 10,
          pileLabelHeight: 4,
          pileLabelTextColor: '#ffffff'
        });
      } else {
        piling.set({
          pileLabel: item => item[label] || '',
          pileLabelText: true,
          pileLabelFontSize: 14,
          pileLabelHeight: 10,
          pileLabelTextColor: '#ffffff'
        });
      }
    }
  };

  return { piling, actions };
}
