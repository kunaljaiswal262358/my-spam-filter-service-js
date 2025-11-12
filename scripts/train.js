const fs = require('fs');
const path = require('path');
const { NaiveBayesClassifier } = require('../src/model/nb_scratch');
const { BaselineClassifier } = require('../src/model/baseline');
const {
  loadAllData,
  stratifiedSplit,
  stratifiedKFold,
  getMetrics,
} = require('../src/utils/data-helpers');

const MODELS_DIR = path.resolve(__dirname, '../models');
const TEST_DATA_PATH = path.resolve(MODELS_DIR, 'test_data.json');
const MODEL_A_PATH = path.resolve(MODELS_DIR, 'nb_scratch_model.json');
const MODEL_B_PATH = path.resolve(MODELS_DIR, 'baseline_model.json');

fs.mkdirSync(MODELS_DIR, { recursive: true });

console.log('Loading data...');
const allData = loadAllData();

console.log('Performing 80/20 stratified train/test split...');
const { trainSet, testSet } = stratifiedSplit(allData, 0.2);

fs.writeFileSync(TEST_DATA_PATH, JSON.stringify(testSet, null, 2));
console.log(`Test set saved to ${TEST_DATA_PATH} (${testSet.length} items)\n`);

const trainDocs = trainSet.map(d => d.text);
const trainLabels = trainSet.map(d => d.label);

console.log('Starting hyperparameter tuning for Model A (alpha)...');
const alphas = [0.1, 0.5, 1.0, 1.5];
let bestAlpha = alphas[0];
let bestF1 = 0;

for (const alpha of alphas) {
  console.log(`  Testing alpha = ${alpha.toFixed(1)}`);
  const folds = stratifiedKFold(trainSet, 5);
  let f1Scores = [];

  for (const { train, test } of folds) {
    const kfClassifier = new NaiveBayesClassifier({ alpha });
    kfClassifier.train(train.map(d => d.text), train.map(d => d.label));
    
    const predictions = test.map(d => kfClassifier.predict(d.text).label);
    const actuals = test.map(d => d.label);
    
    const metrics = getMetrics(predictions, actuals);
    f1Scores.push(metrics.spam.f1);
  }
  
  const avgF1 = f1Scores.reduce((acc, f1) => acc + f1, 0) / f1Scores.length;
  console.log(`    Avg. Spam F1: ${avgF1.toFixed(4)}`);
  
  if (avgF1 > bestF1) {
    bestF1 = avgF1;
    bestAlpha = alpha;
  }
}

console.log(`\nBest alpha selected: ${bestAlpha} (Avg. F1: ${bestF1.toFixed(4)})\n`);

console.log('Training final models on full 80% training set...');

console.log('Training Model A (From Scratch) with best alpha...');
const finalModelA = new NaiveBayesClassifier({ alpha: bestAlpha });
finalModelA.train(trainDocs, trainLabels);
finalModelA.saveModel(MODEL_A_PATH);
console.log(`Model A saved to ${MODEL_A_PATH}`);

console.log('Training Model B (Baseline)...');
const finalModelB = new BaselineClassifier();
finalModelB.train(trainDocs, trainLabels);

finalModelB.saveModel(MODEL_B_PATH)
  .then(() => {
    console.log(`Model B saved to ${MODEL_B_PATH}`);
    console.log('\n--- Training Complete ---');
  })
  .catch(err => {
    console.error('Failed to save Model B:', err);
  });
