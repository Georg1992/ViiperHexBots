# Generated mob descriptors

Built by:

```powershell
.\scripts\build-mob-descriptor.ps1 -Mob <name> -Force
```

Each mob folder contains:

```
generated_descriptors/<mob>/
  simple/
    descriptor.json
```

Source files: `assets/<mob>/<mob>.spr` and `.act` (not tracked, add as needed).
The bot UI uses these descriptor folders as the mob catalog.
