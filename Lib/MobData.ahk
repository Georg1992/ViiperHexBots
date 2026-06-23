; Descriptor-backed mob catalog.
; A mob is available to the bot only when generated_descriptors/<mob>/simple/descriptor.json exists.

global MobNames := []
global MobFolderNames := []

SyncMobDescriptorsFromAssets() {
    assetsRoot := A_ScriptDir . "\assets\mobs"
    buildScript := A_ScriptDir . "\scripts\build-mob-descriptor.ps1"
    descriptorRoot := A_ScriptDir . "\generated_descriptors"

    if (!FileExist(assetsRoot))
        return

    Loop, Files, %assetsRoot%\*, D
    {
        mobName := A_LoopFileName
        sprPath := A_LoopFileFullPath . "\" . mobName . ".spr"
        actPath := A_LoopFileFullPath . "\" . mobName . ".act"
        descriptorPath := descriptorRoot . "\" . mobName . "\simple\descriptor.json"

        needsBuild := false
        if (!FileExist(descriptorPath)) {
            needsBuild := true
        } else {
            FileRead, descriptorText, %descriptorPath%
            if (ErrorLevel != 0 || !InStr(descriptorText, """dead"":")) {
                needsBuild := true
            }
        }
        if (!needsBuild)
            continue
        if (!FileExist(sprPath) && !FileExist(actPath))
            continue
        if (!FileExist(sprPath) || !FileExist(actPath)) {
            MsgBox, 16, ViiperHexBots, Incomplete mob assets for "%mobName%".`n`nExpected:`n%sprPath%`n%actPath%
            continue
        }
        if (!FileExist(buildScript)) {
            MsgBox, 16, ViiperHexBots, Cannot build descriptor for "%mobName%".`n`nMissing script:`n%buildScript%
            continue
        }

        command := "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ . buildScript . """ -Mob """ . mobName . """ -Force"
        RunWait, %command%, %A_ScriptDir%, Hide UseErrorLevel
        if (ErrorLevel != 0) {
            MsgBox, 16, ViiperHexBots, Descriptor generation failed for "%mobName%".`n`nRun manually:`n.\scripts\build-mob-descriptor.ps1 -Mob %mobName%
            continue
        }
    }
}

LoadMobDescriptors() {
    global MobNames, MobFolderNames

    MobNames := []
    MobFolderNames := []
    descriptorRoot := A_ScriptDir . "\generated_descriptors"

    Loop, Files, %descriptorRoot%\*, D
    {
        folderName := A_LoopFileName
        descriptorPath := A_LoopFileFullPath . "\simple\descriptor.json"
        if (!FileExist(descriptorPath))
            continue

        MobFolderNames.Push(folderName)
        MobNames.Push(MobDisplayName(folderName))
    }
}

MobDisplayName(folderName) {
    displayName := StrReplace(folderName, "_", " ")
    displayName := StrReplace(displayName, "-", " ")
    firstChar := SubStr(displayName, 1, 1)
    StringUpper, firstChar, firstChar
    return firstChar . SubStr(displayName, 2)
}

SyncMobDescriptorsFromAssets()
LoadMobDescriptors()
