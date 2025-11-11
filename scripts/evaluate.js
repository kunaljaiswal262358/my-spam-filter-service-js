const path = require('path');
const fs = require('fs');
const { NaiveBayesClassifier } = require('../src/model/nb_scratch');
const { BaselineClassifier } = require('../src/model/baseline');

const MODELS_DIR = path.resolve(__dirname, '../models');
const TEST_SET_PATH = path.resolve(MODELS_DIR, 'test_data.json');
const NB_SCRATCH_PATH = path.resolve(MODELS_DIR, 'nb_scratch_model.json');
const BASELINE_PATH = path.resolve(MODELS_DIR, 'baseline_model.json');

console.log('Loading test data and trained models...');

const testSet = JSON.parse(fs.readFileSync(TEST_SET_PATH, 'utf-8'));
// Manually create the testDocs and testLabels arrays from the testSet
const testDocs = testSet.map(d => d.text);
const testLabels = testSet.map(d => d.label);

// Load Model A (Scratch)
const modelA = NaiveBayesClassifier.loadModel(NB_SCRATCH_PATH);

// Load Model B (Baseline)
let modelB;
let modelBLoaded = false;

BaselineClassifier.loadModel(BASELINE_PATH)
  .then((loadedModel) => {
    modelB = loadedModel;
    modelBLoaded = true;
    console.log('All models loaded. Starting evaluation...\n');
    runEvaluation();
  })
  .catch((err) => {
    // This catch block might catch errors from runEvaluation() too
    console.error('An error occurred during model loading or evaluation:', err);
  });

// --- 2. Evaluation Metrics Helper ---
function calculateMetrics(predictions, trueLabels) {
  let tp = 0; // Spam predicted as Spam (True Positive)
  let fp = 0; // Ham predicted as Spam (False Positive)
  let fn = 0; // Spam predicted as Ham (False Negative)
  let tn = 0; // Ham predicted as Ham (True Negative)
  let misclassified = [];

  for (let i = 0; i < trueLabels.length; i++) {
    const pred = predictions[i];
    const actual = trueLabels[i];
    
    if (actual === 'spam') {
      if (pred === 'spam') tp++;
      else fn++;
    } else { // actual === 'ham'
      if (pred === 'spam') fp++;
      else tn++;
    }

    if (pred !== actual && misclassified.length < 10) {
      misclassified.push({
        // Use the original testDocs array to get the text
        text: testDocs[i].substring(0, 80) + '...',
        predicted: pred,
        actual: actual,
      });
    }
  }

  const accuracy = (tp + tn) / (tp + tn + fp + fn);
  // Metrics for the "spam" class
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = (2 * precision * recall) / (precision + recall) || 0;

  return {
    accuracy,
    precision,
    recall,
    f1,
    confusion: { tp, fp, fn, tn },
    misclassified,
  };
}

// --- 3. Feature Insight Helper ---
function getTopSpamTokens(classifier) {
  const spamLikelihoods = classifier.logLikelihoods.spam;
  const hamLikelihoods = classifier.logLikelihoods.ham;
  const spamminess = {};

  for (const token of classifier.vocabulary) {
    // Find words that are *much* more likely in spam than in ham
    spamminess[token] = spamLikelihoods[token] - (hamLikelihoods[token] || -Infinity);
  }

  return Object.entries(spamminess)
    .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
    .slice(0, 10)
    .map(([token, score]) => token);
}

// --- 4. Main Evaluation Function ---
function runEvaluation() {
  if (!modelBLoaded) return;

  console.log(`Evaluating on ${testDocs.length} held-out test messages...`);

  const predictionsA = testDocs.map(doc => modelA.predict(doc).label);
  const predictionsB = testDocs.map(doc => modelB.predict(doc).label);

  const metricsA = calculateMetrics(predictionsA, testLabels);
  const metricsB = calculateMetrics(predictionsB, testLabels);

  // --- Print Human-Readable Report ---
  console.log('\n--- Model A (From Scratch) ---');
  console.log(`Overall Accuracy: ${(metricsA.accuracy * 100).toFixed(2)}%`);
  console.log('Spam Class Performance:');
  console.log(`  - Precision: ${metricsA.precision.toFixed(4)}`);
  console.log(`  - Recall:    ${metricsA.recall.toFixed(4)}`);
  console.log(`  - F1-Score:  ${metricsA.f1.toFixed(4)}`);
  console.log('Confusion Matrix:');
  console.log(`  - Correctly found spam (True Positives): ${metricsA.confusion.tp}`);
  console.log(`  - Incorrectly called ham 'spam' (False Positives): ${metricsA.confusion.fp}`);
  console.log(`  - Missed spam, called it 'ham' (False Negatives): ${metricsA.confusion.fn}`);
  console.log(`  - Correctly found ham (True Negatives): ${metricsA.confusion.tn}`);


  console.log('\n--- Model B (Library Baseline) ---');
  console.log(`Overall Accuracy: ${(metricsB.accuracy * 100).toFixed(2)}%`);
  console.log('Spam Class Performance:');
  console.log(`  - Precision: ${metricsB.precision.toFixed(4)}`);
  console.log(`  - Recall:    ${metricsB.recall.toFixed(4)}`);
  console.log(`  - F1-Score:  ${metricsB.f1.toFixed(4)}`);
  console.log('Confusion Matrix:');
  console.log(`  - Correctly found spam (True Positives): ${metricsB.confusion.tp}`);
  console.log(`  - Incorrectly called ham 'spam' (False Positives): ${metricsB.confusion.fp}`);
  console.log(`  - Missed spam, called it 'ham' (False Negatives): ${metricsB.confusion.fn}`);
  console.log(`  - Correctly found ham (True Negatives): ${metricsB.confusion.tn}`);

  console.log('\n--- Misclassified Examples (Model A) ---');
  for (const item of metricsA.misclassified) {
    console.log(`  - Predicted: "${item.predicted}", Actual: "${item.actual}", Text: "${item.text}"`);
  }
  
  console.log('\n--- Feature Insight (Top Spam Tokens) ---');
  const topSpamWords = getTopSpamTokens(modelA);
  console.log(`The words that most strongly predict spam are:`);
  console.log(topSpamWords.join(', '));
}
