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

export default async function create(element, items, pileDragEnd) {
    const imageSize = 32;

    const piling = createPilingJs(element, {
        renderer: createImageRenderer({ imageSize }),
        items: items,
        // pileBorderSize: 2,
        // pileOpacity: 0.5 //opaciy piles for the sequence dataset
        pileItemOffset: (item, i, pile) => {
            const isNotLast = pile.items.length - 1 !== i;
            return [+isNotLast * (Math.random() * 12 - 6), +isNotLast * (Math.random() * 12 - 6)];
        },
        pileItemRotation: (item, i, pile) => {
            const isNotLast = pile.items.length - 1 !== i;
            return +isNotLast * (Math.random() * 12 - 6);
        },
        pileSizeBadge: pile => pile.items.length > 1
    });
    piling.arrangeBy('data', ['x', 'y']);
    piling.groupBy('overlap');
    piling.subscribe('pileDragEnd', pileDragEnd);
    return piling;
}
