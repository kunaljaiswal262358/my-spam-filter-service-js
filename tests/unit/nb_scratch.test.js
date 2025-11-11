const { NaiveBayesClassifier } = require('../../src/model/nb_scratch');

// This is our "toy corpus" as required by the spec
// A tiny dataset where we can predict the math
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

  // Before any tests run, train a single classifier on our toy data
  beforeAll(() => {
    classifier = new NaiveBayesClassifier({ alpha: 1.0 }); // Use alpha=1 for simple math
    classifier.train(trainDocs, trainLabels);
  });

  // Test 1: Check if priors (probability of spam vs ham) are correct
  it('should calculate log priors correctly', () => {
    // We have 6 docs total: 3 spam, 3 ham
    // The probability of each class is 3/6 = 0.5
    // We test the log(0.5)
    const expectedLogPrior = Math.log(0.5);
    expect(classifier.logPriors.spam).toBeCloseTo(expectedLogPrior);
    expect(classifier.logPriors.ham).toBeCloseTo(expectedLogPrior);
  });

  // Test 2: Check if likelihoods (probability of a word given a class) are correct
  it('should calculate log likelihoods correctly (with alpha=1)', () => {
    // P(win | spam) = (count('win' in spam) + alpha) / (total spam words + total unique words * alpha)
    // count('win' in spam) = 2
    // total spam words = 9
    // total unique words = 11
    // P = (2 + 1) / (9 + 11 * 1) = 3 / 20
    const expectedLogLike = Math.log(3 / 20);
    expect(classifier.logLikelihoods.spam['win']).toBeCloseTo(expectedLogLike);

    // P(hello | ham) = (count('hello' in ham) + 1) / (total ham words + 11 * 1)
    // count('hello' in ham) = 2
    // total ham words = 10
    // P = (2 + 1) / (10 + 11) = 3 / 21
    const expectedLogLikeHam = Math.log(3 / 21);
    expect(classifier.logLikelihoods.ham['hello']).toBeCloseTo(
      expectedLogLikeHam,
    );
  });

  // Test 3: Check if it can classify an obvious spam message
  it('should correctly classify obvious spam', () => {
    const { label } = classifier.predict('win free prize');
    expect(label).toBe('spam');
  });

  // Test 4: Check if it can classify an obvious ham message
  it('should correctly classify obvious ham', () => {
    const { label } = classifier.predict('hello you today');
    expect(label).toBe('ham');
  });

  // Test 5: Check that it doesn't crash when it sees new words
  it('should handle words not in the vocabulary', () => {
    const { label } = classifier.predict('a new unknown word');
    // With no known words, scores will be equal (just the priors),
    // and our model defaults to 'ham' in a tie. This test just checks that it doesn't crash.
    expect(label).toBe('ham');
  });
});
