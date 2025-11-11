// --- 1. Imports ---
// We import our tools: express, zod, and our NaiveBayesClassifier
const express = require('express');
const path = require('path');
const { z } = require('zod');
const { NaiveBayesClassifier } = require('./model/nb_scratch');

// This line loads the .env file (if it exists)
require('dotenv').config();

// --- 2. Load Model (This is important) ---
// We do this ONCE when the server starts, not for every request.
console.log('Loading the model, please wait...');
const MODEL_PATH = path.resolve(__dirname, '../models/nb_scratch_model.json');
const classifier = NaiveBayesClassifier.loadModel(MODEL_PATH);
console.log('Model loaded successfully.');

// --- 3. Server Setup ---
const app = express();
app.use(express.json()); // This middleware is required to read JSON from requests

// Get the port from the .env file, or default to 3000
const PORT = process.env.PORT || 3000;
// Create the metadata object required by the spec
const modelMetadata = {
  type: 'Multinomial Naive Bayes (From Scratch)',
  version: process.env.MODEL_VERSION || '1.0.0',
  train_date: process.env.MODEL_TRAIN_DATE || 'unknown',
};

// --- 4. Request Logging (PII Safe) ---
// This is a simple logger that runs for every request.
// Per the spec, we log the request but NOT the message content
app.use((req, res, next) => {
  const startTime = process.hrtime();
  // We attach a 'finish' listener to the response
  res.on('finish', () => {
    // When the request is done, we calculate how long it took
    const totalTime = process.hrtime(startTime);
    const totalTimeInMs = (totalTime[0] * 1000 + totalTime[1] / 1e6).toFixed(2);

    // We do NOT log req.body to protect PII
    console.log(
      `${new Date().toISOString()} - ${req.method} ${req.originalUrl} - ${res.statusCode} [${totalTimeInMs}ms]`,
    );
  });

  next(); // This passes the request to the next handler (e.g., /health or /predict)
});

// --- 5. API Endpoints ---

// This is the GET /health endpoint, as required
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'spam-filter-service',
    model: modelMetadata,
  });
});

// This is the validation "rule" using Zod
// It says we expect an object: { messages: ["string", "string", ...] }
const predictSchema = z.object({
  messages: z.array(z.string().min(1)),
});

// This is the POST /predict endpoint, as required
app.post('/predict', (req, res) => {
  // 1. Validate Input
  const validation = predictSchema.safeParse(req.body);

  // 2. Handle Bad Input
  if (!validation.success) {
    // If the input is bad (e.g., not an array), send a 400 error
    return res.status(400).json({
      error: 'Invalid input',
      details: validation.error.issues,
    });
  }

  // 3. Run Predictions
  const { messages } = validation.data;
  const predictions = messages.map((msg) => {
    const result = classifier.predict(msg);
    // Return the label and score, as required
    return {
      label: result.label,
      score: result.scores.spam, // The raw log-probability score for "spam"
    };
  });

  // 4. Send Response
  res.status(200).json({
    predictions: predictions,
  });
});

// --- 6. Error Handling ---
// A simple catch-all error handler for any unexpected crashes
app.use((err, req, res, next) => {
  console.error('An unexpected error occurred:', err);
  res.status(500).json({
    error: 'Internal Server Error',
    message: 'An unexpected error occurred. Please try again later.',
  });
});

// --- 7. Start Server ---
// This line actually starts the server
app.listen(PORT, () => {
  console.log(`Spam filter service listening on http://localhost:${PORT}`);
});
