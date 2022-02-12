import createPilingJs, { createImageRenderer } from 'piling.js';

export default async function create(element, items) {
    return createPilingJs(element, {
        renderer: createImageRenderer(),
        items: items
    });
}
