const { BayesClassifier } = require('natural');
const { preprocess } = require('./nb_scratch');

class BaselineClassifier {
  constructor() {
    this.model = new BayesClassifier();
  }

  train(docs, labels) {
    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      const label = labels[i];

      const tokens = preprocess(doc);
      this.model.addDocument(tokens, label);
    }

    this.model.train();
  }

  predict(doc) {
    const tokens = preprocess(doc);
    const label = this.model.classify(tokens);
    const classifications = this.model.getClassifications(tokens);

    let spamScore = 0;
    const spamClassification = classifications.find((c) => c.label === 'spam');
    if (spamClassification) {
      spamScore = spamClassification.value;
    }

    return {
      label: label,
      score: spamScore,
    };
  }

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
