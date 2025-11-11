const { parse } = require('csv-parse/sync');
const fs = require('fs');
const path = require('path');

const csvFilePath = path.resolve(__dirname, '../data/sms_spam.csv');
const fileContent = fs.readFileSync(csvFilePath, { encoding: 'utf-8' });
const records = parse(fileContent, {
  columns: false,
  from_line: 2,
});

const MINIMAL_STOP_WORDS = new Set(['the', 'is', 'at', 'and', 'a', 'in', 'it']);

function preprocess(text) {
  if (!text) return [];

  let cleanedText = text
    .toLowerCase()
    .replace(/(\r\n|\n|\r)/gm, ' ') // Remove line breaks
    .replace(/(https|http)?:\/\/[^\s]+/g, ' ') // Remove URLs
    .replace(/[£$€]/g, ' price ') // Handle currency symbols
    .replace(/[0-9]+/g, ' number ') // Replace digits with 'number'
    .replace(/[^a-z\s]/g, ' ') // Remove punctuation
    .replace(/\s+/g, ' '); // Normalize whitespace

  return cleanedText
    .split(' ')
    .filter((word) => word.length > 1 && !MINIMAL_STOP_WORDS.has(word));
}

const hamWords = {};
const spamWords = {};
const hamLengths = [];
const spamLengths = [];

for (const record of records) {
  const label = record[0];
  const text = record[1];
  const tokens = preprocess(text);

  if (label === 'ham') {
    hamLengths.push(text ? text.length : 0);
    for (const token of tokens) {
      hamWords[token] = (hamWords[token] || 0) + 1;
    }
  } else if (label === 'spam') {
    spamLengths.push(text ? text.length : 0);
    for (const token of tokens) {
      spamWords[token] = (spamWords[token] || 0) + 1;
    }
  }
}

function getTopWords(wordMap) {
  return Object.entries(wordMap)
    .sort(([, countA], [, countB]) => countB - countA)
    .slice(0, 10)
    .map(([word, count]) => `${word} (${count})`);
}

function getAvgLength(lengths) {
  const total = lengths.reduce((acc, len) => acc + len, 0);
  return (total / lengths.length).toFixed(2);
}

console.log('EDA Insights');

console.log('\nInsight 1: Top 10 Most Frequent Tokens');
console.log('Spam Messages:');
console.log(`  ${getTopWords(spamWords).join(', ')}`);
console.log('\nHam Messages:');
console.log(`  ${getTopWords(hamWords).join(', ')}`);

console.log('\nInsight 2: Message Length Distribution');
console.log(`  Avg. Spam Length: ${getAvgLength(spamLengths)} characters`);
console.log(`  Avg. Ham Length:  ${getAvgLength(hamLengths)} characters`);
