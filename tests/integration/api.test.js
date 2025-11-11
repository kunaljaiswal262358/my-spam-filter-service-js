const request = require('supertest');
const express = require('express');
const path = require('path');
const { z } = require('zod');
const { NaiveBayesClassifier } = require('../../src/model/nb_scratch');

// --- Create a Test App ---
// Instead of importing the *running* app from 'src/app.js',
// we build a *test* version of it here. This is a common and safe pattern.

function createTestApp() {
  require('dotenv').config(); // Load .env variables
  const app = express();
  app.use(express.json());

  // Load the REAL trained model, just like our server does
  const MODEL_PATH = path.resolve(
    __dirname,
    '../../models/nb_scratch_model.json',
  );
  const classifier = NaiveBayesClassifier.loadModel(MODEL_PATH);

  // Get metadata from .env
  const modelMetadata = {
    type: 'Multinomial Naive Bayes (From Scratch)',
    version: process.env.MODEL_VERSION || '1.0.0',
    train_date: process.env.MODEL_TRAIN_DATE || 'unknown',
  };

  // --- We re-create the same routes from src/app.js ---

  // 1. The /health route
  app.get('/health', (req, res) => {
    res.status(200).json({ status: 'ok', model: modelMetadata });
  });

  // 2. The /predict route
  const predictSchema = z.object({
    messages: z.array(z.string().min(1)),
  });

  app.post('/predict', (req, res) => {
    const validation = predictSchema.safeParse(req.body);
    if (!validation.success) {
      // Send 400 error if input is bad
      return res.status(400).json({ error: 'Invalid input' });
    }

    const { messages } = validation.data;
    // Run predictions
    const predictions = messages.map((msg) => {
      const result = classifier.predict(msg);
      return { label: result.label, score: result.scores.spam };
    });

    // Send 200 OK with the predictions
    res.status(200).json({ predictions: predictions });
  });

  return app;
}

// --- Now, we write the tests ---

describe('API Integration Tests', () => {
  let app;

  beforeAll(() => {
    // Create the test server once before all tests
    app = createTestApp();
  });

  // Test 1: Check if the /health endpoint works
  it('GET /health should return 200 and model metadata', async () => {
    const res = await request(app).get('/health');

    expect(res.statusCode).toBe(200);
    expect(res.body.status).toBe('ok');
    expect(res.body.model.type).toBe('Multinomial Naive Bayes (From Scratch)');
    expect(res.body.model.version).toBe(process.env.MODEL_VERSION); // Check if .env loaded
  });

  // Test 2: Check for bad input (empty request)
  it('POST /predict should return 400 for invalid input (e.g., empty body)', async () => {
    const res = await request(app).post('/predict').send({});
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBe('Invalid input');
  });

  // Test 3: Check for bad input (wrong data type)
  it('POST /predict should return 400 for invalid schema (e.g., wrong type)', async () => {
    const res = await request(app)
      .post('/predict')
      .send({ messages: 'this is not an array' });
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBe('Invalid input');
  });

  // Test 4: Check a valid, successful prediction
  it('POST /predict should return 200 and predictions for valid input', async () => {
    const payload = {
      messages: [
        'congratulations you won a free prize call now', // Obvious Spam
        'Hey are you around for dinner tonight?', // Obvious Ham
      ],
    };

    // Send the payload to our test server
    const res = await request(app).post('/predict').send(payload);

    // Check the response
    expect(res.statusCode).toBe(200);
    expect(res.body.predictions).toBeDefined();
    expect(res.body.predictions).toHaveLength(2);

    // Check if the model predicted correctly
    expect(res.body.predictions[0].label).toBe('spam');
    expect(res.body.predictions[1].label).toBe('ham');

    // Check that scores are included
    expect(res.body.predictions[0].score).toBeDefined();
    expect(typeof res.body.predictions[0].score).toBe('number');
  });
});
