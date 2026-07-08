# Mob Recognition (legacy path)

The detection pipeline lives in **`pybot/recognition/`**.

Tests: `tests/recognition/`

## Commands

```powershell
mob-detect build-simple-descriptor --mob horn --force
mob-detect detect-simple --mob horn --image pybot/recognition/test-fixtures/game-screenshots/333.png --debug
mob-detect fixtures-simple --mob horn --debug
```

Or via module:

```powershell
python -m pybot.recognition detect-simple --mob horn --image pybot/recognition/test-fixtures/game-screenshots/333.png --debug
```
