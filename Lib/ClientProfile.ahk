#Include %A_ScriptDir%\TabletLib\JSON.ahk

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

ApplyMemoryDependentUI() {
    global memoryReadingEnabled, clientSupportsMemory, captchaEnabled, captchaLabel
    global warperCoordsSet, TakeFlyWings

    if (!clientSupportsMemory) {
        memoryReadingEnabled := false
        GuiControl,, UseMemoryReading, 0
        GuiControl, Disable, UseMemoryReading
    } else {
        GuiControl, Enable, UseMemoryReading
        if (memoryReadingEnabled)
            GuiControl,, UseMemoryReading, 1
        else
            GuiControl,, UseMemoryReading, 0
    }

    memActive := MemoryFeaturesActive()
    ctlState := memActive ? "Enable" : "Disable"

    GuiControl, %ctlState%, SetWarperCoordsBtn
    GuiControl, % (memActive && warperCoordsSet) ? "Enable" : "Disable", ResetWarperBtn
    GuiControl, %ctlState%, SavePointButtonKey
    GuiControl, %ctlState%, OpenStorageButtonKey
    GuiControl, %ctlState%, WeightModifier
    GuiControl, %ctlState%, WeightModifierText
    GuiControl, %ctlState%, WeightSliderLabel
    GuiControl, %ctlState%, TakeFlyWings
    GuiControl, % (memActive && TakeFlyWings) ? "Enable" : "Disable", FlyWingsAmount

    if (captchaEnabled && memActive) {
        GuiControl, Show, DetectCaptcha
        GuiControl, Enable, DetectCaptcha
        GuiControl,, DetectCaptcha, Detect Captcha (%captchaLabel%)
    } else {
        GuiControl,, DetectCaptcha, 0
        DetectCaptcha := 0
        GuiControl, Hide, DetectCaptcha
    }

    if (memActive && warperCoordsSet) {
        GuiControl, Show, TimeOnLocationTextLabel
        GuiControl, Show, TimeOnLocation
        GuiControl, Show, TimeOnLocationValueText
        GuiControl, Enable, TimeOnLocation
    } else {
        GuiControl, Hide, TimeOnLocationTextLabel
        GuiControl, Hide, TimeOnLocation
        GuiControl, Hide, TimeOnLocationValueText
        GuiControl, Disable, TimeOnLocation
    }
}
