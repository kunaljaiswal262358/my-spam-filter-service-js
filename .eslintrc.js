module.exports = {
  env: {
    node: true,
    commonjs: true,
    es2021: true,
    jest: true, // This line tells ESLint to allow Jest's global keywords (e.g., 'describe', 'it')
  },
  extends: [
    'eslint:recommended',
    'prettier', // This turns off any ESLint rules that conflict with Prettier
  ],
  parserOptions: {
    ecmaVersion: 12,
  },
  rules: {
    // You can add custom rules here
    'no-console': 'off', // We allow console.log for this exercise
  },
};
