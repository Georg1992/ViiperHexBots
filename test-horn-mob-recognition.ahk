#SingleInstance Force
#Include MobData.ahk
#Include utilityFunctions.ahk
#Include Lib\MobRecognition.ahk

CoordMode, Mouse, Screen
CoordMode, Pixel, Screen

global gameWindowID := 0
global gameWindowTitle := ""
global gameProcess := ""
global selectedMonsterIndex := 2
global SearchRange := 16
global cellSize := 50

LoadHornTestConfig()

tip := "Horn Mob Recognition Test (V2)"
tip .= "`nF7 = use active window as game client (recommended)"
tip .= "`nF8 = show hunt search box (red overlay)"
tip .= "`nF9 = find Horn + on-screen hint"
tip .= "`nF10 = same as F9 + debug MsgBox + move mouse"
tip .= "`nEsc = exit"
TrayTip, Horn Mob Recognition Test, %tip%, 8

F7::
    WinGet, gameWindowID, ID, A
    if (!gameWindowID) {
        MsgBox, 48, Horn Test, No active window.
        return
    }
    WinGetTitle, gameWindowTitle, ahk_id %gameWindowID%
    WinGet, gameProcess, ProcessName, ahk_id %gameWindowID%
    SoundBeep, 600, 100
    TrayTip, Horn Test, Game window:`n%gameWindowTitle%, 3
return

F8::
    if (!EnsureGameWindow())
        return
    GetHuntSearchRegion(xs, ys, ws, hs)
    ShowSearchRegionOverlay(xs, ys, ws, hs, 4000)
    ShowMobSearchHint("Search region shown (red box)", 2500, "")
    TrayTip, Horn Test, Red box = OpenCV search area, 3
return

F9::
    RunHornMobSearchTest(false)
return

F10::
    RunHornMobSearchTest(true)
return

Esc::
    Gui, MobHint:Destroy
    ExitApp

LoadHornTestConfig() {
    global gameWindowID, gameWindowTitle, gameProcess, selectedMonsterIndex, SearchRange

    configPath := A_ScriptDir . "\config.ini"
    if (!FileExist(configPath))
        return

    IniRead, gameWindowID, %configPath%, Window, ID, 0
    IniRead, gameWindowTitle, %configPath%, Window, Title,
    IniRead, gameProcess, %configPath%, Window, Process,
    IniRead, selectedMonsterIndex, %configPath%, MonsterSettings, SelectedMonster, 2
    IniRead, SearchRange, %configPath%, Settings, SearchRange, 16
}

EnsureGameWindow() {
    global gameWindowID

    if (gameWindowID && WinExist("ahk_id " gameWindowID))
        return true

    MsgBox, 48, Horn Test, No game window.`n`nFocus your client and press F7, or start main.ahk once so config.ini has a window ID.
    return false
}

BuildRegionReport(xs, ys, ws, hs) {
    global gameWindowID

    WinGetPos, wx, wy, ww, wh, ahk_id %gameWindowID%
    ControlGetPos, cx, cy, cw, ch, , ahk_id %gameWindowID%
    x2 := xs + ws
    y2 := ys + hs

    report := "Detection: simple descriptor heatmap"
    report .= "`nMonster: Horn (index 2)"
    report .= "`nSearch box: " . xs . "," . ys . " to " . x2 . "," . y2
    report .= "`nSize: " . ws . "x" . hs . " (full game client)"
    report .= "`nWindow: " . wx . "," . wy . " " . ww . "x" . wh
    report .= "`nClient:  " . cx . "," . cy . " " . cw . "x" . ch
    return report
}

CountHornTemplates() {
    mobName := MobTemplateFolderName()
    count := 0
    Loop, Files, %A_ScriptDir%\generated_descriptors\%mobName%\simple\templates\*.png
        count++
    return count
}

ParseJsonNumber(jsonText, key, defaultValue := "") {
    if (RegExMatch(jsonText, "i)""" . key . """:([0-9.]+)", match))
        return match1 + 0
    return defaultValue
}

ParseJsonError(jsonText) {
    if (RegExMatch(jsonText, "i)""error"":""([^""]+)""", match))
        return match1
    return ""
}

CountJsonCandidates(jsonText) {
    start := InStr(jsonText, """candidates"":")
    if (!start)
        return 0
    chunk := SubStr(jsonText, start)
    end := InStr(chunk, "]")
    if (!end)
        return 0
    chunk := SubStr(chunk, 1, end)
    count := 0
    pos := 1
    while (pos := RegExMatch(chunk, "i)""templateName"":""", _, pos))
        count++, pos += 1
    return count
}

SummarizeOpenCvResult(jsonText) {
    if (jsonText = "")
        return "OpenCV: no response (Python/cli failure)"

    if (!MobJsonIsOk(jsonText)) {
        err := ParseJsonError(jsonText)
        return "OpenCV: failed" . (err != "" ? " - " . err : "")
    }

    loaded := ParseJsonNumber(jsonText, "templatesLoaded", 0)
    threshold := ParseJsonNumber(jsonText, "confidenceThreshold", 0)
    candidateCount := CountJsonCandidates(jsonText)
    summary := "OpenCV: ok, templates=" . loaded . ", threshold=" . threshold . ", candidates=" . candidateCount

    foundX := 0
    foundY := 0
    conf := 0
    if (MobRecognitionBestCandidate(jsonText, foundX, foundY, conf)) {
        templateName := ""
        if (RegExMatch(jsonText, "i)""best"":\{[^}]*""templateName"":""([^""]+)""", match))
            templateName := match1
        summary .= "`nBest: " . foundX . "," . foundY . " conf=" . conf
        if (templateName != "")
            summary .= " tpl=" . templateName
    } else {
        summary .= "`nBest: none"
    }

    return summary
}

ShowMobSearchHint(message, durationMs := 3500, status := "") {
    Gui, MobHint:Destroy

    bgColor := "2D2D2D"
    if (status = "found")
        bgColor := "1B6B2F"
    else if (status = "miss")
        bgColor := "8B2E2E"
    else if (status = "search")
        bgColor := "2A4A7A"

    Gui, MobHint:+AlwaysOnTop -Caption +ToolWindow +E0x20
    Gui, MobHint:Color, %bgColor%
    Gui, MobHint:Font, s13 bold cFFFFFF
    Gui, MobHint:Add, Text, w420 Center, %message%

    hintX := (A_ScreenWidth - 420) // 2
    if (hintX < 0)
        hintX := 20
    Gui, MobHint:Show, x%hintX% y36 h44 NA

    if (durationMs > 0)
        SetTimer, MobHintHideTimer, -%durationMs%
}

MobHintHideTimer:
    Gui, MobHint:Destroy
return

RunHornMobSearchTest(openCvDebug := false) {
    if (!EnsureGameWindow())
        return

    pythonCmd := EnsureMobRecognitionPython()
    templateCount := CountHornTemplates()
    cliPath := A_ScriptDir . "\" . mobRecognitionCli
    preflight := "Python: " . (pythonCmd != "" ? pythonCmd : "NOT FOUND")
    preflight .= "`nCLI: " . (FileExist(cliPath) ? "ok" : "MISSING")
    preflight .= "`nHorn descriptor templates: " . templateCount

    if (pythonCmd = "" || !FileExist(cliPath)) {
        MsgBox, 48, Horn Test, %preflight%`n`nRun:`n.\mob-recognition\setup.ps1
        return
    }

    if (templateCount = 0) {
        MsgBox, 48, Horn Test, %preflight%`n`nNo descriptor found.`nBuild with:`npy -3 mob-recognition\cli.py build-simple-descriptor --mob horn
        return
    }

    RestoreWindow()
    Sleep, 300

    GetHuntSearchRegion(xs, ys, ws, hs)
    ShowSearchRegionOverlay(xs, ys, ws, hs, 4000)
    ShowMobSearchHint("Searching for Horn... 0s", 0, "search")
    TrayTip, Horn Test, Searching (usually 2-8 seconds)..., 2

    mobName := MobTemplateFolderName()
    jsonText := MobRecognitionDetect(mobName, xs, ys, ws, hs, openCvDebug, true)
    openCvSummary := SummarizeOpenCvResult(jsonText)

    GetMobSearchPlayerIgnore(xs, ys, ws, hs, ignoreX, ignoreY, ignoreW, ignoreH)
    foundX := 0
    foundY := 0
    conf := 0
    found := MobRecognitionBestCandidateFiltered(jsonText, foundX, foundY, conf, ignoreX, ignoreY, ignoreW, ignoreH)
    rawX := 0
    rawY := 0
    rawConf := 0
    rawFound := MobRecognitionBestCandidate(jsonText, rawX, rawY, rawConf)
    resultSummary := found ? ("Match at " . foundX . "," . foundY . " conf=" . conf) : "No match"

    if (!found) {
        SoundBeep, 400, 300
        hint := "Horn NOT FOUND"
        if (InStr(openCvSummary, "no response") || InStr(openCvSummary, "failed"))
            hint .= "`n(OpenCV error — check Python)"
        else if (rawFound)
            hint .= "`nMatch on player (ignored)"
        else
            hint .= "`nNo match in search box"
        ShowMobSearchHint(hint, 4500, "miss")
        TrayTip, Horn Test, NOT FOUND — no Horn in search area, 4

        if (openCvDebug) {
            msg := preflight
            msg .= "`n`n" . BuildRegionReport(xs, ys, ws, hs)
            msg .= "`n`n" . openCvSummary
            msg .= "`n" . resultSummary
            msg .= "`n`nDebug PNG: mob-recognition\debug"
            msg .= "`n`nTip: F7 game window, F8 red box, Horn inside box."
            MsgBox, 48, Horn Test - Not Found, %msg%
        }
        return
    }

    if (openCvDebug)
        MoveMouseTo(foundX, foundY)
    SoundBeep, 800, 150

    hint := "Horn FOUND"
    hint .= "`n" . foundX . ", " . foundY . "  (" . Round(conf * 100) . "%)"
    ShowMobSearchHint(hint, 4500, "found")
    trayFound := "FOUND at " . foundX . ", " . foundY . " (conf " . conf . ")"
    TrayTip, Horn Test, %trayFound%, 4

    if (openCvDebug) {
        msg := "Horn found at " . foundX . ", " . foundY
        msg .= "`n`n" . preflight
        msg .= "`n`n" . BuildRegionReport(xs, ys, ws, hs)
        msg .= "`n`n" . openCvSummary
        msg .= "`n" . resultSummary
        msg .= "`n`nDebug PNG: mob-recognition\debug"
        MsgBox, 64, Horn Test - Found, %msg%
    }
}
