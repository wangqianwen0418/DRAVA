import createPilingJs, { createImageRenderer } from 'piling.js';

export default async function create(element, items) {
    const piling = createPilingJs(element, {
        renderer: createImageRenderer(),
        items: items
    });
    piling.arrangeBy('data', ['x', 'y']);
    piling.groupBy('overlap');
    return piling;
}
