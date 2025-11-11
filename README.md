This is a lightweight Node.js service that classifies SMS-like text as "spam" or "ham" (not spam) using a "from-scratch" Multinomial Naive Bayes classifier.

This project was built to satisfy the "Fresh Joiner Technical Exercise" and includes the model, a training pipeline, and a production-ready REST API.

## Project Structure

```
project/
├── data/
│   └── sms_spam.csv        # The raw training data
├── models/
│   ├── nb_scratch_model.json # Trained "from-scratch" model
│   ├── baseline_model.json   # Trained "baseline" model
│   └── test_data.json      # Held-out 20% test set
├── scripts/
│   ├── train.js            # Runs tuning and trains final models
│   └── evaluate.js         # Runs evaluation on the test set
├── src/
│   ├── model/
│   │   ├── nb_scratch.js   # "From-scratch" Naive Bayes
│   │   └── baseline.js     # "natural" library baseline
│   ├── utils/
│   │   └── data-helpers.js # Data loading & splitting
│   └── app.js              # The Express API server
├── tests/
│   ├── unit/               # Unit tests for model math
│   └── integration/        # Integration tests for the API
├── .env                    # Local environment variables
├── .env.example            # Example environment file
├── package.json
├── jest.config.js
└── REPORT.md               # Full analysis and evaluation report
```

## Prerequisites

- Node.js (v18.x or v20.x)
- npm

## Installation

1.  Clone this repository.
2.  Install all required dependencies:
    ```bash
    npm install
    ```

## Running the Service

There are three main parts to this project: Training, Evaluating, and Running the API.

### 1. Training the Model

This script will load the raw `data/sms_spam.csv`, perform an 80/20 split, run 5-fold cross-validation on the training set to find the best `alpha`, and then save the final trained models and the test set to the `/models` directory.

```bash
npm run train
```

### 2. Evaluating the Model

This script loads the held-out test set and the trained models from the `/models` directory. It runs a full evaluation and prints the accuracy, precision, recall, F1-score, and confusion matrix.

```bash
npm run evaluate
```

### 3. Running the API Server

This command starts the live API server using the trained "from-scratch" model (`nb_scratch_model.json`).

```bash
npm start
```

The server will be running at `http://localhost:3000`.

## Running Tests

This project includes unit tests for the classifier's math and integration tests for the API endpoints.

```bash
npm test
```

## API Endpoints

### `GET /health`

Returns the service status and model metadata.

**Response:**

```json
{
  "status": "ok",
  "service": "spam-filter-service",
  "model": {
    "type": "Multinomial Naive Bayes (From Scratch)",
    "version": "1.0.0",
    "train_date": "2025-11-10"
  }
}
```

### `POST /predict`

Returns spam/ham predictions for a batch of messages.

**Request Body:**

```json
{
  "messages": ["free entry to win a prize", "hey are you free tonight?"]
}
```

**Success Response (200):**

```json
{
  "predictions": [
    {
      "label": "spam",
      "score": -11.982
    },
    {
      "label": "ham",
      "score": -20.431
    }
  ]
}
```

**Error Response (400):**

```json
{
  "error": "Invalid input",
  "details": [
    /* ... zod error details ... */
  ]
}
```
