module.exports = {
    parser: '@typescript-eslint/parser',
    plugins: ['prettier', 'jest'],
    extends: [
        'eslint:recommended',
        'plugin:react/recommended',
        'plugin:jest/recommended',
        'plugin:jest/style',
        'prettier',
        'plugin:prettier/recommended'
    ],
    env: {
        browser: true,
        node: true
    },
    settings: {
        react: {
            version: 'detect'
        }
    },
    rules: {
        // https://github.com/jest-community/eslint-plugin-jest#rules
        'prettier/prettier': 'warn',
        // https://github.com/typescript-eslint/typescript-eslint/tree/master/packages/eslint-plugin#supported-rules
        'no-explicit-any': 'off',
        'no-empty-interface': 'off',
        'explicit-module-boundary-types': 'off',
        'no-namespace': 'off',
        'ban-ts-comment': 'off',
        'no-unused-vars': 'off',
        // https://eslint.org/docs/rules/
        'linebreak-style': ['error', 'unix'],
        'no-console': ['error', { allow: ['warn', 'error', 'info'] }],
        'no-trailing-spaces': 'off',
        'no-irregular-whitespace': ['error', { skipComments: true }],
        'no-alert': 'error',
        'prefer-const': 'warn',
        'no-case-declarations': 'warn',
        'no-return-assign': 'error',
        'no-useless-call': 'error',
        'no-useless-concat': 'error',
        'prefer-template': 'error'
    }
};
