const { BayesClassifier } = require('natural');
// We re-use the *same* preprocess function so it's a fair comparison
const { preprocess } = require('./nb_scratch');

// This is our "wrapper" class for the library's model.
// It gives it the same 'train' and 'predict' functions as our scratch model.
class BaselineClassifier {
  constructor() {
    this.model = new BayesClassifier();
  }

  // --- The "learning" function ---
  train(docs, labels) {
    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      const label = labels[i];

      const tokens = preprocess(doc);
      // We just add documents one by one
      this.model.addDocument(tokens, label);
    }

    // The library handles all the math internally
    this.model.train();
  }

  // --- The "guessing" function ---
  predict(doc) {
    const tokens = preprocess(doc);
    // The library calculates the prediction
    const label = this.model.classify(tokens);
    // It can also give us the raw scores
    const classifications = this.model.getClassifications(tokens);

    let spamScore = 0;
    const spamClassification = classifications.find((c) => c.label === 'spam');
    if (spamClassification) {
      spamScore = spamClassification.value;
    }

    return {
      label: label,
      score: spamScore, // This is a probability score (e.g., 0.98)
    };
  }

  // --- Functions to save and load the trained model ---
  saveModel(path) {
    return new Promise((resolve, reject) => {
      this.model.save(path, (err) => {
        if (err) return reject(err);
        resolve();
      });
    });
  }

  static loadModel(path) {
    return new Promise((resolve, reject) => {
      BayesClassifier.load(path, null, (err, classifier) => {
        if (err) return reject(err);
        const modelWrapper = new BaselineClassifier();
        modelWrapper.model = classifier;
        resolve(modelWrapper);
      });
    });
  }
}

module.exports = { BaselineClassifier };
