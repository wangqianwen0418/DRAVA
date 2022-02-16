import createPilingJs from 'piling.js';

/**
 * Promised-based image loading
 * @param   {string}  src  Remote image source, i.e., a URL
 * @return  {object}  Promise resolving to the image once its loaded
 */
export const loadImage = (src, isCrossOrigin = false) =>
    new Promise((resolve, reject) => {
        const image = new Image(64, 64);
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

const createImageRenderer = () => sources =>
    Promise.all(
        sources.map(src => {
            const isCrossOrigin = true;
            return loadImage(src, isCrossOrigin);
        })
    );

export default async function create(element, items, pileDragEnd) {
    const piling = createPilingJs(element, {
        renderer: createImageRenderer(),
        items: items
    });
    piling.arrangeBy('data', ['x', 'y']);
    piling.groupBy('overlap');
    piling.subscribe('pileDragEnd', pileDragEnd);
    return piling;
}
