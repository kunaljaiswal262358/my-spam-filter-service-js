const request = require('supertest');
const express = require('express');
const path = require('path');
const { z } = require('zod');
const { NaiveBayesClassifier } = require('../../src/model/nb_scratch');


function createTestApp() {
  require('dotenv').config(); 
  const app = express();
  app.use(express.json());

  const MODEL_PATH = path.resolve(
    __dirname,
    '../../models/nb_scratch_model.json',
  );
  const classifier = NaiveBayesClassifier.loadModel(MODEL_PATH);

  const modelMetadata = {
    type: 'Multinomial Naive Bayes (From Scratch)',
    version: process.env.MODEL_VERSION || '1.0.0',
    train_date: process.env.MODEL_TRAIN_DATE || 'unknown',
  };

  app.get('/health', (req, res) => {
    res.status(200).json({ status: 'ok', model: modelMetadata });
  });

  const predictSchema = z.object({
    messages: z.array(z.string().min(1)),
  });

  app.post('/predict', (req, res) => {
    const validation = predictSchema.safeParse(req.body);
    if (!validation.success) {
      return res.status(400).json({ error: 'Invalid input' });
    }

    const { messages } = validation.data;
    
    const predictions = messages.map((msg) => {
      const result = classifier.predict(msg);
      return { label: result.label, score: result.scores.spam };
    });

    res.status(200).json({ predictions: predictions });
  });

  return app;
}

describe('API Integration Tests', () => {
  let app;

  beforeAll(() => {
    app = createTestApp();
  });

  it('GET /health should return 200 and model metadata', async () => {
    const res = await request(app).get('/health');
    const MODEL_VERSION = process.env.MODEL_VERSION || '1.0.0';

    expect(res.statusCode).toBe(200);
    expect(res.body.status).toBe('ok');
    expect(res.body.model.type).toBe('Multinomial Naive Bayes (From Scratch)');
    expect(res.body.model.version).toBe(MODEL_VERSION);
  });

  it('POST /predict should return 400 for invalid input (e.g., empty body)', async () => {
    const res = await request(app).post('/predict').send({});
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBe('Invalid input');
  });

  it('POST /predict should return 400 for invalid schema (e.g., wrong type)', async () => {
    const res = await request(app)
      .post('/predict')
      .send({ messages: 'this is not an array' });
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBe('Invalid input');
  });

  it('POST /predict should return 200 and predictions for valid input', async () => {
    const payload = {
      messages: [
        'congratulations you won a free prize call now',
        'Hey are you around for dinner tonight?',
      ],
    };

    const res = await request(app).post('/predict').send(payload);

    expect(res.statusCode).toBe(200);
    expect(res.body.predictions).toBeDefined();
    expect(res.body.predictions).toHaveLength(2);

    expect(res.body.predictions[0].label).toBe('spam');
    expect(res.body.predictions[1].label).toBe('ham');

    expect(res.body.predictions[0].score).toBeDefined();
    expect(typeof res.body.predictions[0].score).toBe('number');
  });
});
