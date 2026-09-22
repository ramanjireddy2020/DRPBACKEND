# InnoDD AWS Lambda Proxy

Thin Lambda proxy in front of the InnoDD API
(`https://innodd-api-7474645714257046.aws.databricksapps.com`), so a frontend
calling this Lambda (via API Gateway) never has to deal with Databricks' two
auth layers directly.

`backend_client.py` holds both auth layers:

1. **Databricks Apps' own OAuth gate** — client-credentials token for the app's
   service principal (`app-2yztdm innodd-api`), cached and auto-refreshed.
2. **The DRP `/v1` API's own login session** — sent as `X-Drp-Token` (not
   `Authorization`, since layer 1 owns that header), refreshed via a 30-day
   refresh token rather than re-sending the password on every call.

`lambda_function.py` is the Lambda entry point (`lambda_handler`) — it just
extracts method/path/body/params from the API Gateway event and calls
`call_backend(method, path, json_body, params)`, the one generic function that
reaches all 62 of InnoDD's `/v1/*` endpoints (full method+path reference list
is in `backend_client.py`'s module docstring). Nothing needs updating here when
a new `/v1` endpoint is added on the backend side.

## Files

| File | Purpose |
|---|---|
| `backend_client.py` | Auth (both layers) + `call_backend()` — the actual proxying logic |
| `lambda_function.py` | Lambda entry point (`lambda_handler`) — the file/handler name AWS assumes by default |
| `requirements.txt` | Just `httpx` |

## Required environment variables

| Variable | Value |
|---|---|
| `DATABRICKS_TOKEN_URL` | `https://dbc-96626831-6c96.cloud.databricks.com/oidc/v1/token` |
| `DATABRICKS_CLIENT_ID` | `83535854-5015-4b8c-a6c9-0e2b048f0f3c` |
| `DATABRICKS_CLIENT_SECRET` | the app's service-principal OAuth secret (valid until 2028-08-23) |
| `INNODD_BASE_URL` | `https://innodd-api-7474645714257046.aws.databricksapps.com` |
| `DRP_EMAIL` | `priya@drp.app` (demo account — swap for a real one before production) |
| `DRP_PASSWORD` | `drp-demo-password` |

**Set these via AWS Secrets Manager / SSM Parameter Store references, not plain
Lambda environment variables** — `DATABRICKS_CLIENT_SECRET` in particular is a
live, long-lived credential.

## Deploy

```
pip install -r requirements.txt -t package/
cp backend_client.py lambda_function.py package/
cd package && zip -r ../proxy.zip .
aws lambda create-function --function-name innodd-proxy \
  --runtime python3.12 --handler lambda_function.lambda_handler \
  --zip-file fileb://../proxy.zip --role <lambda-execution-role-arn>
```

Put API Gateway (HTTP API or REST API — the handler detects either payload
format automatically) in front of it, with a `{proxy+}` route so every `/v1/*`
path forwards through to this Lambda.
