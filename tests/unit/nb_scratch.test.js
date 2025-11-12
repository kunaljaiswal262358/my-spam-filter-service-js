const { NaiveBayesClassifier } = require('../../src/model/nb_scratch');

const trainDocs = [
  'win prize money',
  'claim free prize',
  'prize win free',
  'hello how are you',
  'are you free today',
  'hello friend',
];
const trainLabels = ['spam', 'spam', 'spam', 'ham', 'ham', 'ham'];

describe('NaiveBayesClassifier (Unit Tests)', () => {
  let classifier;

  beforeAll(() => {
    classifier = new NaiveBayesClassifier({ alpha: 1.0 }); 
    classifier.train(trainDocs, trainLabels);
  });

  it('should calculate log priors correctly', () => {
    const expectedLogPrior = Math.log(0.5);
    expect(classifier.logPriors.spam).toBeCloseTo(expectedLogPrior);
    expect(classifier.logPriors.ham).toBeCloseTo(expectedLogPrior);
  });

  it('should calculate log likelihoods correctly (with alpha=1)', () => {
    const expectedLogLike = Math.log(3 / 20);
    expect(classifier.logLikelihoods.spam['win']).toBeCloseTo(expectedLogLike);

    const expectedLogLikeHam = Math.log(3 / 21);
    expect(classifier.logLikelihoods.ham['hello']).toBeCloseTo(
      expectedLogLikeHam,
    );
  });

  it('should correctly classify obvious spam', () => {
    const { label } = classifier.predict('win free prize');
    expect(label).toBe('spam');
  });

  it('should correctly classify obvious ham', () => {
    const { label } = classifier.predict('hello you today');
    expect(label).toBe('ham');
  });

  it('should handle words not in the vocabulary', () => {
    const { label } = classifier.predict('a new unknown word');
    expect(label).toBe('ham');
  });
});
