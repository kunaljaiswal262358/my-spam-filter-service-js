const path = require('path');
const fs = require('fs');
const { NaiveBayesClassifier } = require('../src/model/nb_scratch');
const { BaselineClassifier } = require('../src/model/baseline');
const {
  loadData,
  stratifiedTrainTestSplit,
  stratifiedKFold,
} = require('../src/utils/data-helpers');

const MODELS_DIR = path.resolve(__dirname, '../models');
const TEST_SET_PATH = path.resolve(MODELS_DIR, 'test_data.json');
const NB_SCRATCH_PATH = path.resolve(MODELS_DIR, 'nb_scratch_model.json');
const BASELINE_PATH = path.resolve(MODELS_DIR, 'baseline_model.json');

// --- 1. Load and Split Data ---
console.log('Loading data...');
const { docs, labels } = loadData();

console.log('Performing 80/20 stratified train/test split...');
const { trainDocs, trainLabels, testDocs, testLabels } =
  stratifiedTrainTestSplit(docs, labels, 0.2);

// Save test set for Task 4
fs.writeFileSync(TEST_SET_PATH, JSON.stringify({ testDocs, testLabels }));
console.log(`Test set saved to ${TEST_SET_PATH} (${testDocs.length} items)`);

// --- 2. Hyperparameter Tuning (Model A) ---
console.log('\nStarting hyperparameter tuning for Model A (alpha)...');
const alphas = [0.1, 0.5, 1.0, 1.5]; // Hyperparameter grid
const kFolds = 5;
let bestAlpha = 1.0;
let bestAvgF1 = -1;

for (const alpha of alphas) {
  console.log(`  Testing alpha = ${alpha}`);
  let foldF1Scores = [];
  const kFoldIterator = stratifiedKFold(trainDocs, trainLabels, kFolds);

  for (const fold of kFoldIterator) {
    const classifier = new NaiveBayesClassifier({ alpha: alpha });
    classifier.train(fold.trainDocs, fold.trainLabels);

    let spamTruePositive = 0;
    let spamPredictedPositive = 0;
    let spamActualPositive = 0;

    fold.valLabels.forEach((trueLabel, i) => {
      const pred = classifier.predict(fold.valDocs[i]).label;
      if (trueLabel === 'spam') {
        spamActualPositive++;
        if (pred === 'spam') {
          spamTruePositive++;
        }
      }
      if (pred === 'spam') {
        spamPredictedPositive++;
      }
    });

    const precision =
      spamPredictedPositive === 0
        ? 0
        : spamTruePositive / spamPredictedPositive;
    const recall =
      spamActualPositive === 0 ? 0 : spamTruePositive / spamActualPositive;
    const f1 =
      precision + recall === 0
        ? 0
        : (2 * (precision * recall)) / (precision + recall);
    foldF1Scores.push(f1);
  }

  const avgF1 = foldF1Scores.reduce((a, b) => a + b, 0) / kFolds;
  console.log(`    Avg. Spam F1: ${avgF1.toFixed(4)}`);

  if (avgF1 > bestAvgF1) {
    bestAvgF1 = avgF1;
    bestAlpha = alpha;
  }
}

console.log(
  `\nBest alpha selected: ${bestAlpha} (Avg. F1: ${bestAvgF1.toFixed(4)})`,
);

// --- 3. Final Model Training ---
console.log('\nTraining final models on full 80% training set...');

// Train Model A (Scratch)
console.log('Training Model A (From Scratch) with best alpha...');
const finalModelA = new NaiveBayesClassifier({ alpha: bestAlpha });
finalModelA.train(trainDocs, trainLabels);
finalModelA.saveModel(NB_SCRATCH_PATH);
console.log(`Model A saved to ${NB_SCRATCH_PATH}`);

// Train Model B (Baseline)
console.log('Training Model B (Baseline)...');
const finalModelB = new BaselineClassifier();
finalModelB.train(trainDocs, trainLabels);
finalModelB
  .saveModel(BASELINE_PATH)
  .then(() => {
    console.log(`Model B saved to ${BASELINE_PATH}`);
    console.log('\n--- Training Complete ---');
  })
  .catch((err) => {
    console.error('Error saving Model B:', err);
  });
