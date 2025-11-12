## API Endpoints
run the server using 
```bash
npm run start
```
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
    /* ... These are zod error details on user passed data... */
  ]
}
```
