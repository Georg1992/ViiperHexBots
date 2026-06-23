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
py -3 mob-recognition\cli.py build-simple-descriptor --mob horn
```

Output goes to `generated_descriptors/<mob>/simple/` (descriptor, templates, masks, accents, audit).
