# Apple Health Bridge Companion

Minimal iOS companion app for manually syncing confirmed nutrition writes from the backend into Apple Health.

## First Version Scope

- Request HealthKit write authorization
- Configure backend base URL, bridge token, and user id
- Tap `Sync Now`
- Pull `/integrations/apple-health/pending-writes`
- Write nutrition samples into Apple Health
- Report `/integrations/apple-health/write-result`

Not in scope yet:

- HealthKit reads
- background sync
- retries
- multi-device coordination

## Project

- Xcode project: `AppleHealthBridgeCompanion.xcodeproj`
- App target: `AppleHealthBridgeCompanion`

## Capability Setup

Open the target in Xcode and enable:

1. `Signing & Capabilities`
2. Add `HealthKit`
3. Select your Apple Development team for signing

The project already includes:

- `AppleHealthBridgeCompanion.entitlements`
- `NSHealthUpdateUsageDescription`
- `NSHealthShareUsageDescription`

## Backend Configuration

Launch the app on a real iPhone and fill:

- `Base URL`
  Example: `http://192.168.1.23:8000`
- `Bridge Token`
  Match `APPLE_HEALTH_BRIDGE_TOKEN` from the backend if set
- `User ID`
  Example: `tg:123456`

## Backend Expectations

The backend must expose:

- `POST /integrations/apple-health/pending-writes`
- `POST /integrations/apple-health/write-result`

The current app expects the existing payload shape from `app/nutrition/apple_health_mapping.py`.

## Real Device Test

1. Start the backend.
   Recommended for LAN testing:
   `uvicorn main:app --host 0.0.0.0 --port 8000`
2. Produce a confirmed meal log draft.
3. Install the app on a real iPhone from Xcode.
4. Put the iPhone and Mac on the same Wi-Fi.
5. Find the Mac LAN IP, for example:
   `ipconfig getifaddr en0`
6. Enter `Base URL` as `http://<mac-lan-ip>:8000`
7. Tap `Request Health Access`.
8. Tap `Sync Now`.
9. Verify:
   - backend logs show `pending_writes`
   - backend logs show `write_result`
   - the app shows the sync summary
   - Apple Health shows the dietary nutrition samples

## ATS Note

This first version temporarily enables arbitrary network loads so the app can call a local `http://<lan-ip>:8000` backend during device testing.
Tighten ATS before shipping a production build.

## Local Build Check

From the repo root:

```bash
xcodebuild \
  -project ios/AppleHealthBridgeCompanion/AppleHealthBridgeCompanion.xcodeproj \
  -scheme AppleHealthBridgeCompanion \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath tmp/xcode-derived \
  CODE_SIGNING_ALLOWED=NO \
  build
```
