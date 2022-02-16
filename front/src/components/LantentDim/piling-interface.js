import createPilingJs, { createImageRenderer } from 'piling.js';

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
