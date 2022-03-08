import createPilingJs from 'piling.js';

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
  const imageSize = 64;
  const { items, pileDragEnd, dims, getSvgGroup, dataset } = pilingOptions;

  var spec;

  if (dataset == 'sequence') {
    spec = {
      renderer: createImageRenderer({ imageSize }),
      items: items,
      itemSize: imageSize,
      pileItemOpacity: (item, i, pile) => 1 - i / pile.items.length, //opaciy piles for the sequence dataset
      pileSizeBadge: pile => pile.items.length > 1,
      pileItemOffset: [0, 0] //force all items overlaid
    };
  } else if (dataset == 'dsprites') {
    spec = {
      renderer: createImageRenderer({ imageSize }),
      items: items,
      itemSize: imageSize,
      pileItemOpacity: (item, i, pile) => 1 - i / pile.items.length, //opaciy piles for the dsprites dataset
      pileSizeBadge: pile => pile.items.length > 1,
      pileItemOffset: [0, 0] //force all items overlaid
    };
  } else {
    spec = {
      renderer: createImageRenderer({ imageSize }),
      items: items,
      itemSize: imageSize,
      // items in a pile is randomly rotated and offset
      pileItemOffset: (item, i, pile) => {
        const isNotLast = pile.items.length - 1 !== i;
        return [+isNotLast * (Math.random() * 12 - 6), +isNotLast * (Math.random() * 12 - 6)];
      },
      pileItemRotation: (item, i, pile) => {
        const isNotLast = pile.items.length - 1 !== i;
        return +isNotLast * (Math.random() * 12 - 6);
      },
      pileSizeBadge: pile => pile.items.length > 1
    };
  }

  const piling = createPilingJs(element, spec);

  piling.arrangeBy('data', dims);

  piling.subscribe('pileDragEnd', pileDragEnd);
  piling.subscribe('zoom', camera => {
    const svgGroup = getSvgGroup();
    svgGroup.attr(
      'transform',
      `translate(${camera.translation[0]}, 0) scale(${camera.scaling} 1)` // only update translate x and scale x
    );
    // to prevent the distortion of svg elements
    svgGroup.selectAll('image').attr('transform', `scale(${1 / camera.scaling} 1)`);
    svgGroup.selectAll('rect').attr('transform', `scale(${1 / camera.scaling} 1)`);
    svgGroup.selectAll('text').attr('transform', `scale(${1 / camera.scaling} 1)`);
  });

  const actions = {
    reArrange: dims => piling.arrangeBy('data', dims),
    group: dim => piling.groupBy('category', item => item['assignments'][dim]),
    splitAll: () => piling.splitAll()
  };

  return [piling, actions];
}
