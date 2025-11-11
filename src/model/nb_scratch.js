const fs = require('fs');

// --- Part 1: The Preprocessing Function ---
// (Same as before)
// ------------------------------------------
const MINIMAL_STOP_WORDS = new Set(['the', 'is', 'at', 'and', 'a', 'in', 'it']);

function preprocess(text) {
  if (typeof text !== 'string') {
    return [];
  }

  let cleanedText = text
    .toLowerCase()
    .replace(/(\r\n|\n|\r)/gm, ' ')
    .replace(/(https|http)?:\/\/[^\s]+/g, ' ')
    .replace(/[£$€]/g, ' price ')
    .replace(/[0-9]+/g, ' number ')
    .replace(/[^a-z\s]/g, ' ')
    .replace(/\s+/g, ' ');

  return cleanedText
    .split(' ')
    .filter((word) => word.length > 1 && !MINIMAL_STOP_WORDS.has(word));
}

// --- Part 2: The Classifier "Blueprint" ---
// (Same as before)
// ------------------------------------------
class NaiveBayesClassifier {
  constructor(options = {}) {
    this.alpha = options.alpha || 1;
    this.logPriors = {};
    this.logLikelihoods = {};
    // We must convert the Set to an Array to save it as JSON
    this.vocabulary = new Set();
  }

  // --- "Learning" function (Same as before) ---
  train(docs, labels) {
    const classDocs = { ham: [], spam: [] };
    const classWordCounts = { ham: {}, spam: {} };
    const classTotalWordCount = { ham: 0, spam: 0 };
    let totalDocs = docs.length;

    for (let i = 0; i < totalDocs; i++) {
      const label = labels[i];
      const tokens = preprocess(docs[i]);
      classDocs[label].push(tokens);

      for (const token of tokens) {
        this.vocabulary.add(token);
        classWordCounts[label][token] =
          (classWordCounts[label][token] || 0) + 1;
        classTotalWordCount[label]++;
      }
    }

    const vocabSize = this.vocabulary.size;

    for (const docClass of ['ham', 'spam']) {
      const docCountForClass = classDocs[docClass].length;
      this.logPriors[docClass] = Math.log(docCountForClass / totalDocs);

      const totalWordsInClass = classTotalWordCount[docClass];
      const likelihoods = {};

      for (const token of this.vocabulary) {
        const tokenCount = classWordCounts[docClass][token] || 0;
        const numerator = tokenCount + this.alpha;
        const denominator = totalWordsInClass + vocabSize * this.alpha;
        likelihoods[token] = Math.log(numerator / denominator);
      }
      this.logLikelihoods[docClass] = likelihoods;
    }
  }

  // --- "Guessing" function (Same as before) ---
  predict(doc) {
    const tokens = preprocess(doc);
    const scores = { ham: 0, spam: 0 };

    for (const docClass of ['ham', 'spam']) {
      let classScore = this.logPriors[docClass];
      const likelihoods = this.logLikelihoods[docClass];

      for (const token of tokens) {
        // We use the Set for fast lookups
        if (this.vocabulary.has(token)) {
          classScore += likelihoods[token];
        }
      }
      scores[docClass] = classScore;
    }

    const prediction = scores.spam > scores.ham ? 'spam' : 'ham';

    return {
      label: prediction,
      scores: scores,
    };
  }

  // --- [NEW] Function to save the model's brain ---
  saveModel(filePath) {
    const modelData = {
      alpha: this.alpha,
      logPriors: this.logPriors,
      logLikelihoods: this.logLikelihoods,
      // Convert Set to Array for JSON.stringify()
      vocabulary: Array.from(this.vocabulary),
    };

    fs.writeFileSync(filePath, JSON.stringify(modelData, null, 2));
  }

  // --- [NEW] Function to load the model's brain ---
  static loadModel(filePath) {
    const modelData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

    // Create a new classifier and "re-hydrate" it
    const classifier = new NaiveBayesClassifier({ alpha: modelData.alpha });
    classifier.logPriors = modelData.logPriors;
    classifier.logLikelihoods = modelData.logLikelihoods;
    // Convert Array back to Set for fast O(1) lookups
    classifier.vocabulary = new Set(modelData.vocabulary);

    return classifier;
  }
}

module.exports = { NaiveBayesClassifier, preprocess };
