#Requires AutoHotkey v1.1.33+

; Session-based debug log — one file per application launch under logs\sessions\

global sessionLogId := ""
global sessionLogPath := ""
global sessionLogDir := ""
global sessionStartedTick := 0
global sessionBotRunCount := 0

SessionLogSanitize(text) {
    text := StrReplace(text, "`r", " ")
    text := StrReplace(text, "`n", " ")
    return text
}

SessionLogTimestamp() {
    FormatTime, ts,, yyyy-MM-dd HH:mm:ss.fff
    return ts
}

SessionLogEnsureDir() {
    global sessionLogDir
    if (sessionLogDir != "")
        return sessionLogDir

    sessionLogDir := A_ScriptDir . "\logs\sessions"
    if (!FileExist(sessionLogDir))
        FileCreateDir, %sessionLogDir%
    return sessionLogDir
}

SessionLogAppendLine(line) {
    global sessionLogPath
    if (sessionLogPath = "")
        return

    line := SessionLogSanitize(line)
    FileAppend, %line%`n, %sessionLogPath%, UTF-8
}

SessionLogWrite(level, category, message) {
    global sessionLogPath
    if (sessionLogPath = "")
        return

    category := SessionLogSanitize(category)
    message := SessionLogSanitize(message)
    line := "[" . SessionLogTimestamp() . "] [" . level . "]"
    if (category != "")
        line .= " [" . category . "]"
    line .= " " . message
    SessionLogAppendLine(line)
}

SessionLogWriteBlock(title, content) {
    global sessionLogPath
    if (sessionLogPath = "")
        return

    SessionLogAppendLine("")
    SessionLogAppendLine("--- " . title . " ---")
    Loop, Parse, content, `n, `r
    {
        if (A_LoopField != "")
            SessionLogAppendLine("  " . SessionLogSanitize(A_LoopField))
    }
    SessionLogAppendLine("---")
}

SessionLogStart() {
    global sessionLogId, sessionLogPath, sessionStartedTick, sessionBotRunCount

    SessionLogEnsureDir()
    FormatTime, sessionLogId,, yyyyMMdd_HHmmss
    sessionLogPath := sessionLogDir . "\" . sessionLogId . ".log"
    sessionStartedTick := A_TickCount
    sessionBotRunCount := 0

    header := "=== ViiperHexBots debug session ==="
    header .= "`nsessionId: " . sessionLogId
    header .= "`nstartedAt: " . SessionLogTimestamp()
    header .= "`nahkVersion: " . A_AhkVersion
    header .= "`nosVersion: " . A_OSVersion
    header .= "`nscriptDir: " . A_ScriptDir
    header .= "`nscriptName: " . A_ScriptName
    header .= "`nisAdmin: " . (A_IsAdmin ? "yes" : "no")
    header .= "`nscreen: " . A_ScreenWidth . "x" . A_ScreenHeight
    SessionLogWriteBlock("session start", header)

    SessionLogWrite("INFO", "session", "Debug log file: " . sessionLogPath)
}

SessionLogWriteRuntimeContext() {
    global mobRecognitionPython, inputReady, gameWindowID, gameWindowTitle, gameProcess

    lines := ""
    if (mobRecognitionPython != "")
        lines .= "pythonCmd: " . mobRecognitionPython . "`n"
    lines .= "mobDetector: simple-cli`n"
    lines .= "inputReady: " . (inputReady ? "yes" : "no") . "`n"
    if (gameWindowID) {
        lines .= "gameWindowId: " . gameWindowID . "`n"
        lines .= "gameWindowTitle: " . gameWindowTitle . "`n"
        lines .= "gameProcess: " . gameProcess . "`n"
        if (WinExist("ahk_id " . gameWindowID)) {
            WinGetPos, wx, wy, ww, wh, ahk_id %gameWindowID%
            ControlGetPos, cx, cy, cw, ch, , ahk_id %gameWindowID%
            lines .= "windowRect: " . wx . "," . wy . " " . ww . "x" . wh . "`n"
            lines .= "clientRect: " . (wx + cx) . "," . (wy + cy) . " " . cw . "x" . ch . "`n"
            lines .= "gameFocused: " . (WinActive("ahk_id " . gameWindowID) ? "yes" : "no")
        } else {
            lines .= "gameWindowExists: no"
        }
    } else {
        lines .= "gameWindow: not selected"
    }
    SessionLogWriteBlock("runtime context", lines)
}

SessionLogRegisterBotRun() {
    global sessionBotRunCount, gameWindowID, gameWindowTitle, gameProcess
    global SkillButtonKey, TeleportButtonKey, SavePointButtonKey, SkillTimerButtonKey
    global SearchRange, cellSize, selectedMonsterIndex, MobNames
    global botPaused, warperCoordsSet, warperLocation, clientProfileName

    sessionBotRunCount++
    mobName := MobTemplateFolderName()

    GetHuntSearchRegion(xs, ys, ws, hs)
    GetMobSearchPlayerIgnore(xs, ys, ws, hs, ignoreX, ignoreY, ignoreW, ignoreH)

    skillSC := GetKeySC(SkillButtonKey) + 0
    teleportSC := GetKeySC(TeleportButtonKey) + 0

    lines := "botRun: #" . sessionBotRunCount . "`n"
    lines .= "startedAt: " . SessionLogTimestamp() . "`n"
    lines .= "memoryReading: " . (MemoryFeaturesActive() ? "on" : "off") . "`n"
    lines .= "clientProfile: " . clientProfileName . "`n"
    lines .= "monster: " . MobNames[selectedMonsterIndex] . " (" . mobName . ")`n"
    lines .= "searchRange: " . SearchRange . " cells (" . (SearchRange * cellSize) . "px)`n"
    lines .= "huntRoi: " . xs . "," . ys . " " . ws . "x" . hs . "`n"
    lines .= "playerIgnore: " . ignoreX . "," . ignoreY . " " . ignoreW . "x" . ignoreH . "`n"
    lines .= "skillKey: " . SkillButtonKey . " (sc=" . skillSC . ")`n"
    lines .= "teleportKey: " . TeleportButtonKey . " (sc=" . teleportSC . ")`n"
    lines .= "savePointKey: " . SavePointButtonKey . "`n"
    lines .= "skillTimerKey: " . SkillTimerButtonKey . "`n"
    if (warperCoordsSet)
        lines .= "warperLocation: " . warperLocation . "`n"
    lines .= "gameWindowId: " . gameWindowID . "`n"
    lines .= "gameWindowTitle: " . gameWindowTitle . "`n"
    lines .= "gameProcess: " . gameProcess . "`n"
    lines .= "gameFocused: " . (WinActive("ahk_id " . gameWindowID) ? "yes" : "no")

    SessionLogWriteBlock("bot run #" . sessionBotRunCount, lines)
    SessionLogWriteRuntimeContext()
}

SessionLogHuntScan(mobName, xs, ys, ws, hs) {
    global gameWindowID
    focused := (gameWindowID && WinActive("ahk_id " . gameWindowID)) ? "yes" : "no"
    SessionLogWrite("DEBUG", "hunt"
        , "scan mob=" . mobName . " roi=" . xs . "," . ys . " " . ws . "x" . hs . " gameFocused=" . focused)
}

SessionLogDetectResponse(jsonText, elapsedMs) {
    if (jsonText = "") {
        SessionLogWrite("WARN", "detect", "empty response after " . elapsedMs . "ms")
        return
    }

    ok := InStr(jsonText, """ok"":true") ? "yes" : "no"
    detectMs := ""
    if (RegExMatch(jsonText, "i)""detectMs"":(\d+)", m))
        detectMs := m1

    candidateCount := 0
    pos := 1
    while (pos := RegExMatch(jsonText, "i)""accepted"":true", match, pos)) {
        candidateCount++
        pos += StrLen(match)
    }

    bestX := ""
    bestY := ""
    bestConf := ""
    if (RegExMatch(jsonText, "i)""best"":\{[^}]*""centerX"":(\d+)[^}]*""centerY"":(\d+)[^}]*""confidence"":([0-9.]+)", b)) {
        bestX := b1
        bestY := b2
        bestConf := b3
    } else if (RegExMatch(jsonText, "i)""centerX"":(\d+)", bx)) {
        bestX := bx1
        if (RegExMatch(jsonText, "i)""centerY"":(\d+)", by))
            bestY := by1
        if (RegExMatch(jsonText, "i)""confidence"":([0-9.]+)", bc))
            bestConf := bc1
    }

    summary := "ok=" . ok . " elapsed=" . elapsedMs . "ms"
    if (detectMs != "")
        summary .= " detectMs=" . detectMs
    summary .= " accepted=" . candidateCount
    if (bestX != "")
        summary .= " best=" . bestX . "," . bestY . " conf=" . bestConf

    if (!InStr(jsonText, """ok"":true") && RegExMatch(jsonText, "i)""error"":""([^""]+)""", err))
        summary .= " error=" . err1

    SessionLogWrite("DEBUG", "detect", summary)
}

SessionLogFocusChange(event, activeWindowId := "") {
    global gameWindowID
    detail := event
    if (activeWindowId != "")
        detail .= " activeHwnd=" . activeWindowId
    if (gameWindowID)
        detail .= " gameHwnd=" . gameWindowID . " gameActive=" . (WinActive("ahk_id " . gameWindowID) ? "yes" : "no")
    SessionLogWrite("INFO", "focus", detail)
}

SessionLogEnd(reason := "exit") {
    global sessionLogPath, sessionStartedTick, sessionBotRunCount

    if (sessionLogPath = "")
        return

    elapsedSec := (A_TickCount - sessionStartedTick) // 1000
    footer := "reason: " . reason
    footer .= "`nendedAt: " . SessionLogTimestamp()
    footer .= "`nuptimeSec: " . elapsedSec
    footer .= "`nbotRuns: " . sessionBotRunCount
    SessionLogWriteBlock("session end", footer)

    sessionLogPath := ""
}

OnExit("SessionLogOnExit")

SessionLogOnExit(ExitReason, ExitCode) {
    if (sessionLogPath = "")
        return
    SessionLogEnd(ExitReason . " (code " . ExitCode . ")")
}
