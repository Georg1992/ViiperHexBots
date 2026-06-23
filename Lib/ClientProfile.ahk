#Include %A_ScriptDir%\Lib\JSON.ahk

global clientProfileName := "HoneyRO"
global clientSupportsMemory := false
global memoryReadingEnabled := false
global captchaEnabled := false
global captchaColor := 0
global captchaLabel := ""
global zoomWheelDelta := 1
global zoomSteps := 10
global zoomDelayMs := 60

ParseHexAddress(hexValue) {
    if (hexValue = "" || hexValue = "ERROR")
        return 0
    return hexValue + 0
}

ParseHexColor(hexValue) {
    if (hexValue = "" || hexValue = "ERROR")
        return 0
    return hexValue + 0
}

ListClientProfiles() {
    profiles := []
    Loop, Files, %A_ScriptDir%\clients\*.json
        profiles.Push(RegExReplace(A_LoopFileName, "\.json$"))
    return profiles
}

LoadClientProfile(profileName) {
    global clientProfileName, clientSupportsMemory
    global maxSpAddress, currentSpAddress, currentWeightAddress, totalWeightAddress, currentLocationAddress
    global captchaEnabled, captchaColor, captchaLabel
    global zoomWheelDelta, zoomSteps, zoomDelayMs

    profilePath := A_ScriptDir . "\clients\" . profileName . ".json"
    if (!FileExist(profilePath)) {
        MsgBox, 16, ViiperHexBots, Client profile not found:`n%profilePath%
        return false
    }

    FileRead, jsonText, %profilePath%
    profile := JSON.Load(jsonText)
    clientProfileName := profile.name

    maxSpAddress := 0
    currentSpAddress := 0
    currentWeightAddress := 0
    totalWeightAddress := 0
    currentLocationAddress := 0
    clientSupportsMemory := false

    if (IsObject(profile.memory)) {
        maxSpAddress := ParseHexAddress(profile.memory.maxSpAddress)
        currentSpAddress := ParseHexAddress(profile.memory.currentSpAddress)
        currentWeightAddress := ParseHexAddress(profile.memory.currentWeightAddress)
        totalWeightAddress := ParseHexAddress(profile.memory.totalWeightAddress)
        currentLocationAddress := ParseHexAddress(profile.memory.currentLocationAddress)
        clientSupportsMemory := (maxSpAddress && currentWeightAddress && currentLocationAddress)
    }

    captchaEnabled := false
    captchaColor := 0
    captchaLabel := ""
    if (IsObject(profile.captcha)) {
        captchaEnabled := true
        captchaColor := ParseHexColor(profile.captcha.color)
        captchaLabel := profile.captcha.label
    }

    zoomWheelDelta := 1
    zoomSteps := 10
    zoomDelayMs := 60
    if (IsObject(profile.zoom)) {
        if (profile.zoom.wheelDelta != "")
            zoomWheelDelta := profile.zoom.wheelDelta
        if (profile.zoom.steps != "")
            zoomSteps := profile.zoom.steps
        if (profile.zoom.delayMs != "")
            zoomDelayMs := profile.zoom.delayMs
    }

    return true
}

ApplyZoomDirectionFromGlobal() {
    global ZoomWheelDirection, zoomWheelDelta
    if (ZoomWheelDirection = "Scroll down")
        zoomWheelDelta := -1
    else
        zoomWheelDelta := 1
}

MemoryFeaturesActive() {
    global memoryReadingEnabled, clientSupportsMemory
    return memoryReadingEnabled && clientSupportsMemory
}
