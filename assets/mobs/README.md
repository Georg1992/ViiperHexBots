# Mob assets

Place each mob in its own folder:

```
assets/mobs/
  horn/
    horn.spr
    horn.act
  poring/
    poring.spr
    poring.act
```

Build the runtime descriptor:

```powershell
.\scripts\build-mob-descriptor.ps1 -Mob horn -Force
```

Output goes to `generated_descriptors/<mob>/simple/descriptor.json`.
The bot UI loads mobs from those descriptor folders on startup, so new mobs appear there after their descriptor exists.
