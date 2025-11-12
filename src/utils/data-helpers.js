const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');

const csvFilePath = path.resolve(__dirname, '../../data/sms_spam.csv');


function loadAllData() {
  const fileContent = fs.readFileSync(csvFilePath, { encoding: 'utf-8' });
  const records = parse(fileContent, {
    columns: false,
    from_line: 2, // Skip header
  });

  // Transform [ 'ham', 'text...' ] into { label: 'ham', text: 'text...' }
  return records.map(r => ({
    label: r[0],
    text: r[1],
  }));
}


function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}


function stratifiedSplit(data, testSize = 0.2) {
  const ham = data.filter(d => d.label === 'ham');
  const spam = data.filter(d => d.label === 'spam');

  shuffle(ham);
  shuffle(spam);

  const hamTestCount = Math.floor(ham.length * testSize);
  const spamTestCount = Math.floor(spam.length * testSize);

  const testSet = ham.slice(0, hamTestCount).concat(spam.slice(0, spamTestCount));
  const trainSet = ham.slice(hamTestCount).concat(spam.slice(spamTestCount));

  return { trainSet, testSet };
}

/**
 * A generator function for stratified K-Fold cross-validation.
 * @param {Array<{label: string, text: string}>} data The training dataset
 * @param {number} k The number of folds
 */
function* stratifiedKFold(data, k = 5) {
  const ham = data.filter(d => d.label === 'ham');
  const spam = data.filter(d => d.label === 'spam');

  shuffle(ham);
  shuffle(spam);

  const hamFoldSize = Math.floor(ham.length / k);
  const spamFoldSize = Math.floor(spam.length / k);

  for (let i = 0; i < k; i++) {
    const valHamStart = i * hamFoldSize;
    const valSpamStart = i * spamFoldSize;

    // Handle the last fold to include all remaining items
    const valHamEnd = (i === k - 1) ? ham.length : (i + 1) * hamFoldSize;
    const valSpamEnd = (i === k - 1) ? spam.length : (i + 1) * spamFoldSize;

    const test = ham.slice(valHamStart, valHamEnd).concat(spam.slice(valSpamStart, valSpamEnd));
    
    const train = ham.slice(0, valHamStart).concat(ham.slice(valHamEnd))
      .concat(spam.slice(0, valSpamStart)).concat(spam.slice(valSpamEnd));
    
    yield { train, test };
  }
}


function getMetrics(predictions, trueLabels) {
  let tp = 0, fp = 0, fn = 0;

  for (let i = 0; i < trueLabels.length; i++) {
    const pred = predictions[i];
    const actual = trueLabels[i];

    if (actual === 'spam') {
      if (pred === 'spam') tp++;
      else fn++;
    } else { // actual === 'ham'
      if (pred === 'spam') fp++;
    }
  }
  
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = (2 * precision * recall) / (precision + recall) || 0;

  return {
    spam: {
      precision,
      recall,
      f1,
    }
  };
}


module.exports = {
  loadAllData,
  stratifiedSplit,
  stratifiedKFold,
  getMetrics,
};
