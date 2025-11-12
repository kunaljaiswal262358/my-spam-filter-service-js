const express = require('express');
const path = require('path');
const { z } = require('zod');
const { NaiveBayesClassifier } = require('./model/nb_scratch');

require('dotenv').config();

console.log('Loading the model, please wait...');
const MODEL_PATH = path.resolve(__dirname, '../models/nb_scratch_model.json');
const classifier = NaiveBayesClassifier.loadModel(MODEL_PATH);
console.log('Model loaded successfully.');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;
const modelMetadata = {
  type: 'Multinomial Naive Bayes (From Scratch)',
  version: process.env.MODEL_VERSION || '1.0.0',
  train_date: process.env.MODEL_TRAIN_DATE || 'unknown',
};

app.use((req, res, next) => {
  const startTime = process.hrtime();
  res.on('finish', () => {
    const totalTime = process.hrtime(startTime);
    const totalTimeInMs = (totalTime[0] * 1000 + totalTime[1] / 1e6).toFixed(2);

    console.log(
      `${new Date().toISOString()} - ${req.method} ${req.originalUrl} - ${res.statusCode} [${totalTimeInMs}ms]`,
    );
  });

  next();
});

app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'spam-filter-service',
    model: modelMetadata,
  });
});

const predictSchema = z.object({
  messages: z.array(z.string().min(1)),
});

app.post('/predict', (req, res) => {
  const validation = predictSchema.safeParse(req.body);

  if (!validation.success) {
    return res.status(400).json({
      error: 'Invalid input',
      details: validation.error.issues,
    });
  }

  const { messages } = validation.data;
  const predictions = messages.map((msg) => {
    const result = classifier.predict(msg);

    return {
      label: result.label,
      score: result.scores.spam, 
    };
  });

  res.status(200).json({
    predictions: predictions,
  });
});

app.use((err, req, res, next) => {
  console.error('An unexpected error occurred:', err);
  res.status(500).json({
    error: 'Internal Server Error',
    message: 'An unexpected error occurred. Please try again later.',
  });
});

app.listen(PORT, () => {
  console.log(`Spam filter service listening on http://localhost:${PORT}`);
});
