# End-to-end check of the API Gateway path: Cognito login -> gateway -> Lambda -> app.
#
# Prompts for a Cognito password (never echoed, never stored) and runs a few
# endpoints, including ones that only work now the catch-all route exists.
#
# Usage:  .\scripts\test_gateway.ps1  [-Email someone@innominds.com]

param(
    [string]$Email  = "bpuvvada@innominds.com",
    [string]$Client = "4amqg86j1fpf6fcp7cuph1cg8q",
    [string]$Gw     = "https://uk1ip13n80.execute-api.us-east-1.amazonaws.com"
)

$ErrorActionPreference = "Stop"

$secure = Read-Host "Cognito password for $Email" -AsSecureString
$plain  = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
            [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))

$body = @{
    AuthFlow       = "USER_PASSWORD_AUTH"
    ClientId       = $Client
    AuthParameters = @{ USERNAME = $Email; PASSWORD = $plain }
} | ConvertTo-Json -Depth 5

$tmp = Join-Path $env:TEMP "cognito_login.json"
[IO.File]::WriteAllText($tmp, $body)

Write-Host "`n[1] Cognito login" -ForegroundColor Cyan
$resp = & curl.exe -s -H "Content-Type: application/x-amz-json-1.1" `
    -H "X-Amz-Target: AWSCognitoIdentityProviderService.InitiateAuth" `
    -d "@$tmp" "https://cognito-idp.us-east-1.amazonaws.com/"
Remove-Item $tmp -Force
$plain = $null

$json = $resp | ConvertFrom-Json
if (-not $json.AuthenticationResult) {
    Write-Host "   FAILED: $resp" -ForegroundColor Red
    if ($json.__type -eq "NEW_PASSWORD_REQUIRED" -or $json.ChallengeName) {
        Write-Host "   (account must finish its first-time password change in the UI first)" -ForegroundColor Yellow
    }
    exit 1
}
$idToken = $json.AuthenticationResult.IdToken
Write-Host "   OK - idToken acquired (valid 60 min)" -ForegroundColor Green
Write-Host "   paste this into Postman's cognito_token variable (Current value):" -ForegroundColor DarkGray
Write-Host "   $idToken`n" -ForegroundColor DarkGray

function Hit($label, $method, $path, $payload) {
    $args = @("-s", "-o", "$env:TEMP\gwout.json", "-w", "%{http_code}",
              "-X", $method, "-H", "Authorization: Bearer $idToken")
    if ($payload) {
        [IO.File]::WriteAllText("$env:TEMP\gwbody.json", $payload)
        $args += @("-H", "Content-Type: application/json", "-d", "@$env:TEMP\gwbody.json")
    }
    $code = & curl.exe @args "$Gw$path"
    $out  = [IO.File]::ReadAllText("$env:TEMP\gwout.json")
    $head = $out.Substring(0, [Math]::Min(110, $out.Length))
    $colour = if ($code -eq "200" -or $code -eq "201") { "Green" } else { "Red" }
    Write-Host ("   {0,-34} {1}  {2}" -f $label, $code, $head) -ForegroundColor $colour
    return $out
}

Write-Host "[2] Endpoints that already had routes" -ForegroundColor Cyan
Hit "GET /users/me"        "GET"  "/users/me"        $null | Out-Null
Hit "GET /modules"         "GET"  "/modules"         $null | Out-Null
Hit "GET /dashboard/summary" "GET" "/dashboard/summary" $null | Out-Null

Write-Host "`n[3] Endpoints that needed the catch-all route" -ForegroundColor Cyan
$sess = Hit "POST /sessions" "POST" "/sessions" '{"query":"Find targets for thrombocytosis"}'
$sid = ($sess | ConvertFrom-Json).id
if ($sid) {
    Hit "GET /sessions/{id}/artifacts" "GET" "/sessions/$sid/artifacts" $null | Out-Null
    Hit "POST /sessions/{id}/messages" "POST" "/sessions/$sid/messages" '{"message":"Why is JAK2 first?"}' | Out-Null
}

Write-Host "`nIf everything above is 200/201, the collection works for anyone with a Cognito user." -ForegroundColor Cyan
