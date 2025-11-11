const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');

const csvFilePath = path.resolve(__dirname, '../../data/sms_spam.csv');

function loadData() {
  const fileContent = fs.readFileSync(csvFilePath, { encoding: 'utf-8' });
  const records = parse(fileContent, {
    columns: false,
    from_line: 2,
  });

  const docs = records.map((r) => r[1]);
  const labels = records.map((r) => r[0]);

  return { docs, labels };
}

// Standard Fisher-Yates shuffle
function shuffle(data) {
  let { docs, labels } = data;
  let N = docs.length;
  for (let i = N - 1; i > 0; i--) {
    let j = Math.floor(Math.random() * (i + 1));
    [docs[i], docs[j]] = [docs[j], docs[i]];
    [labels[i], labels[j]] = [labels[j], labels[i]];
  }
  return { docs, labels };
}

function stratifiedTrainTestSplit(docs, labels, testSize = 0.2) {
  const hamDocs = [];
  const spamDocs = [];
  const hamLabels = [];
  const spamLabels = [];

  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 'ham') {
      hamDocs.push(docs[i]);
      hamLabels.push(labels[i]);
    } else {
      spamDocs.push(docs[i]);
      spamLabels.push(labels[i]);
    }
  }

  // Shuffle each class independently
  shuffle({ docs: hamDocs, labels: hamLabels });
  shuffle({ docs: spamDocs, labels: spamLabels });

  const hamTestCount = Math.floor(hamDocs.length * testSize);
  const spamTestCount = Math.floor(spamDocs.length * testSize);

  const testDocs = hamDocs
    .slice(0, hamTestCount)
    .concat(spamDocs.slice(0, spamTestCount));
  const testLabels = hamLabels
    .slice(0, hamTestCount)
    .concat(spamLabels.slice(0, spamTestCount));

  const trainDocs = hamDocs
    .slice(hamTestCount)
    .concat(spamDocs.slice(spamTestCount));
  const trainLabels = hamLabels
    .slice(hamTestCount)
    .concat(spamLabels.slice(spamTestCount));

  return { trainDocs, trainLabels, testDocs, testLabels };
}

function* stratifiedKFold(docs, labels, k = 5) {
  const hamDocs = [];
  const spamDocs = [];
  const hamLabels = [];
  const spamLabels = [];

  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 'ham') {
      hamDocs.push(docs[i]);
      hamLabels.push(labels[i]);
    } else {
      spamDocs.push(docs[i]);
      spamLabels.push(labels[i]);
    }
  }

  // Shuffle before folding
  shuffle({ docs: hamDocs, labels: hamLabels });
  shuffle({ docs: spamDocs, labels: spamLabels });

  const hamFoldSize = Math.floor(hamDocs.length / k);
  const spamFoldSize = Math.floor(spamDocs.length / k);

  for (let i = 0; i < k; i++) {
    const valHamStart = i * hamFoldSize;
    const valSpamStart = i * spamFoldSize;

    // Handle the last fold to include all remaining items
    const valHamEnd = i === k - 1 ? hamDocs.length : (i + 1) * hamFoldSize;
    const valSpamEnd = i === k - 1 ? spamDocs.length : (i + 1) * spamFoldSize;

    const valDocs = hamDocs
      .slice(valHamStart, valHamEnd)
      .concat(spamDocs.slice(valSpamStart, valSpamEnd));
    const valLabels = hamLabels
      .slice(valHamStart, valHamEnd)
      .concat(spamLabels.slice(valSpamStart, valSpamEnd));

    const trainDocs = hamDocs
      .slice(0, valHamStart)
      .concat(hamDocs.slice(valHamEnd))
      .concat(spamDocs.slice(0, valSpamStart))
      .concat(spamDocs.slice(valSpamEnd));
    const trainLabels = hamLabels
      .slice(0, valHamStart)
      .concat(hamLabels.slice(valHamEnd))
      .concat(spamLabels.slice(0, valSpamStart))
      .concat(spamLabels.slice(valSpamEnd));

    yield { trainDocs, trainLabels, valDocs, valLabels };
  }
}

module.exports = {
  loadData,
  stratifiedTrainTestSplit,
  stratifiedKFold,
};
