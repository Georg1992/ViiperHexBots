# Generated mob descriptors

Built by:

```powershell
py -3 mob-recognition\cli.py build-simple-descriptor --mob <name>
```

Each mob folder contains:

```
generated_descriptors/<mob>/
  simple/
    descriptor.json
    templates/
    masks/
    accents/
    debug_contact_sheet.png
    descriptor_audit.json
```

Source files: `assets/mobs/<mob>/<mob>.spr` and `.act`
