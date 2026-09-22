# Two AWS changes that make the Postman collection usable by people without
# Databricks accounts. Run once, from a terminal where you are logged in.
#
#   1. API Gateway catch-all route  -> the 7 newest endpoints stop returning 404
#      (steps, messages, rerun, artifacts, curatex results/profile, litminex preview)
#   2. Cognito ALLOW_USER_PASSWORD_AUTH -> Postman's "0 · Cognito Login" can mint
#      an idToken, instead of everyone copying one out of the browser every hour
#
# Both are additive. SRP stays enabled, so the web app is unaffected, and the
# existing per-endpoint gateway routes keep working alongside the catch-all.
#
# Usage:  .\scripts\setup_gateway_access.ps1

$ErrorActionPreference = "Stop"

$API        = "uk1ip13n80"
$INTEGRATION= "integrations/cjsic6e"
$AUTHORIZER = "laga83"
$POOL       = "us-east-1_AYZb6Pkib"
$CLIENT     = "4amqg86j1fpf6fcp7cuph1cg8q"
$env:AWS_DEFAULT_REGION = "us-east-1"

# Reuse the SSO session if one was cached by the earlier device login.
$credFile = Join-Path $env:TEMP "awscreds.json"
if (Test-Path $credFile) {
    $c = Get-Content $credFile | ConvertFrom-Json
    $env:AWS_ACCESS_KEY_ID     = $c.k
    $env:AWS_SECRET_ACCESS_KEY = $c.s
    $env:AWS_SESSION_TOKEN     = $c.t
    Write-Host "Using cached SSO credentials." -ForegroundColor DarkGray
}

Write-Host "`nIdentity:" -ForegroundColor Cyan
aws sts get-caller-identity --output text --query "Arn"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not authenticated. Run: aws configure sso   (or redo the SSO device login)" -ForegroundColor Red
    exit 1
}

Write-Host "`n[1/2] API Gateway catch-all routes" -ForegroundColor Cyan
$routes = aws apigatewayv2 get-routes --api-id $API --max-results 500 --query "Items[].RouteKey" --output json | ConvertFrom-Json

if ($routes -contains 'ANY /{proxy+}') {
    Write-Host "   ANY /{proxy+} already exists - skipping" -ForegroundColor DarkGray
} else {
    aws apigatewayv2 create-route --api-id $API --route-key 'ANY /{proxy+}' `
        --target $INTEGRATION --authorization-type JWT --authorizer-id $AUTHORIZER `
        --query "RouteKey" --output text
    Write-Host "   created ANY /{proxy+}" -ForegroundColor Green
}

if ($routes -contains 'OPTIONS /{proxy+}') {
    Write-Host "   OPTIONS /{proxy+} already exists - skipping" -ForegroundColor DarkGray
} else {
    # No authorizer on OPTIONS: browsers send CORS preflight without an Authorization header.
    aws apigatewayv2 create-route --api-id $API --route-key 'OPTIONS /{proxy+}' `
        --target $INTEGRATION --query "RouteKey" --output text
    Write-Host "   created OPTIONS /{proxy+}" -ForegroundColor Green
}

Write-Host "`n[2/2] Cognito: add ALLOW_USER_PASSWORD_AUTH" -ForegroundColor Cyan
$flows = aws cognito-idp describe-user-pool-client --user-pool-id $POOL --client-id $CLIENT `
    --query "UserPoolClient.ExplicitAuthFlows" --output json | ConvertFrom-Json

if ($flows -contains "ALLOW_USER_PASSWORD_AUTH") {
    Write-Host "   already enabled - skipping" -ForegroundColor DarkGray
} else {
    $new = @($flows) + "ALLOW_USER_PASSWORD_AUTH"
    aws cognito-idp update-user-pool-client --user-pool-id $POOL --client-id $CLIENT `
        --explicit-auth-flows $new --query "UserPoolClient.ExplicitAuthFlows" --output text
    Write-Host "   enabled (SRP kept, so the web app is unaffected)" -ForegroundColor Green
}

Write-Host "`nDone. In Postman: run '0 - Cognito Login' with a pool user's email + password." -ForegroundColor Cyan
Write-Host "Pool currently has one user: bpuvvada@innominds.com" -ForegroundColor DarkGray
Write-Host "To add more:  aws cognito-idp admin-create-user --user-pool-id $POOL --username <email> --user-attributes Name=email,Value=<email> Name=email_verified,Value=true" -ForegroundColor DarkGray
