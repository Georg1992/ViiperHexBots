#Requires AutoHotkey v1.1.33+

; Session logs — one folder per application launch under logs\sessions\

global sessionLogId := ""
global sessionLogPath := ""
global sessionLogDir := ""
global sessionLogRootDir := ""
global sessionBehaviorLogPath := ""
global sessionStartedTick := 0
global sessionBotRunCount := 0

SessionLogSanitize(text) {
    text := StrReplace(text, "`r", " ")
    text := StrReplace(text, "`n", " ")
    return text
}

SessionLogJsonEscape(text) {
    text := StrReplace(text, "\", "\\")
    text := StrReplace(text, """", "'")
    text := StrReplace(text, "`r", " ")
    text := StrReplace(text, "`n", " ")
    return text
}

SessionLogTimestamp() {
    FormatTime, ts,, yyyy-MM-dd HH:mm:ss.fff
    return ts
}

SessionLogEnsureDir() {
    global sessionLogRootDir
    if (sessionLogRootDir = "") {
        sessionLogRootDir := A_ScriptDir . "\logs\sessions"
        if (!FileExist(sessionLogRootDir))
            FileCreateDir, %sessionLogRootDir%
    }
    return sessionLogRootDir
}

SessionLogActiveDir() {
    global sessionLogDir
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

BehaviorLogWrite(message) {
    global sessionBehaviorLogPath
    if (sessionBehaviorLogPath = "")
        return

    message := SessionLogSanitize(message)
    line := "[" . SessionLogTimestamp() . "] " . message
    FileAppend, %line%`n, %sessionBehaviorLogPath%, UTF-8
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

SessionLogPruneOldSessions(keepCount := 3) {
    rootDir := SessionLogEnsureDir()
    sessionNames := ""

    Loop, Files, %rootDir%\*, D
        sessionNames .= A_LoopFileName . "`n"

    Sort, sessionNames, R

    index := 0
    Loop, Parse, sessionNames, `n, `r
    {
        if (A_LoopField = "")
            continue

        index++
        if (index <= keepCount)
            continue

        oldSessionDir := rootDir . "\" . A_LoopField
        FileRemoveDir, %oldSessionDir%, 1
    }
}

SessionLogStart() {
    global sessionLogId, sessionLogPath, sessionLogDir, sessionBehaviorLogPath
    global sessionStartedTick, sessionBotRunCount

    rootDir := SessionLogEnsureDir()
    FormatTime, sessionLogId,, yyyyMMdd_HHmmss
    sessionLogDir := rootDir . "\" . sessionLogId
    if (!FileExist(sessionLogDir))
        FileCreateDir, %sessionLogDir%
    sessionLogPath := sessionLogDir . "\system.log"
    sessionBehaviorLogPath := sessionLogDir . "\behavior.log"
    SessionLogPruneOldSessions(3)
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

    SessionLogWrite("INFO", "session", "System log file: " . sessionLogPath)
    BehaviorLogWrite("Session started")
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

    if (!GetHuntSearchRegion(xs, ys, ws, hs)) {
        xs := 0
        ys := 0
        ws := 0
        hs := 0
    }
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
    global sessionLogPath, sessionBehaviorLogPath, sessionStartedTick, sessionBotRunCount

    if (sessionLogPath = "")
        return

    if IsFunc("BotSessionStop") {
        fn := "BotSessionStop"
        %fn%(reason)
    }

    elapsedSec := (A_TickCount - sessionStartedTick) // 1000
    footer := "reason: " . reason
    footer .= "`nendedAt: " . SessionLogTimestamp()
    footer .= "`nuptimeSec: " . elapsedSec
    footer .= "`nbotRuns: " . sessionBotRunCount
    SessionLogWriteBlock("session end", footer)
    BehaviorLogWrite("Session ended: " . reason)

    sessionLogPath := ""
    sessionBehaviorLogPath := ""
}

OnExit("SessionLogOnExit")

SessionLogOnExit(ExitReason, ExitCode) {
    if (sessionLogPath = "")
        return
    SessionLogEnd(ExitReason . " (code " . ExitCode . ")")
}
