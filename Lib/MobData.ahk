; Descriptor-backed mob catalog.
; A mob is available to the bot only when generated_descriptors/<mob>/simple/descriptor.json exists.

global MobNames := []
global MobFolderNames := []

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

LoadMobDescriptors()
