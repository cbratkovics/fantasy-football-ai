# Deployment environment

The frontend needs exactly one environment variable:

```
NEXT_PUBLIC_API_URL=https://<your-deployed-api-host>
```

Set it to the public URL of the deployed ffai FastAPI service (for example a Hugging Face Space
running `uvicorn ffai.serve.app:app --port 7860`). Every page reads its data from that API at
runtime; there is no auth, payment, database, or other configuration. When unset, the build falls
back to `http://localhost:7860` (see `next.config.js`).
