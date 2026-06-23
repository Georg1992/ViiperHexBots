#Requires AutoHotkey v1.1.33+

; Session-based debug log — one file per application launch under logs\sessions\

global sessionLogId := ""
global sessionLogPath := ""
global sessionLogDir := ""
global sessionStartedTick := 0
global sessionBotRunCount := 0
global botSessionActive := false
global botSessionId := ""
global botSessionDir := ""
global botSessionSummaryPath := ""
global botSessionStartedTick := 0
global botSessionTargetMob := ""
global botSessionTotalScans := 0
global botSessionEmptyScans := 0
global botSessionTargetScans := 0
global botSessionDetectFailures := 0
global botSessionAttacksIssued := 0
global botSessionResolvedEngagements := 0
global botSessionAcceptedCandidates := 0
global botSessionLastRoi := ""
global botSessionScaleMob := ""
global botSessionScaleObservations := ""
global botSessionScaleCount := 0
global botSessionScaleLocked := false
global botSessionScaleMin := 0.0
global botSessionScaleMax := 0.0

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

BotSessionStart(mobName) {
    global sessionBotRunCount, sessionLogId
    global botSessionActive, botSessionId, botSessionDir, botSessionSummaryPath, botSessionStartedTick
    global botSessionTargetMob, botSessionTotalScans, botSessionEmptyScans, botSessionTargetScans
    global botSessionDetectFailures, botSessionAttacksIssued, botSessionResolvedEngagements
    global botSessionAcceptedCandidates, botSessionLastRoi, botSessionScaleMob
    global botSessionScaleObservations, botSessionScaleCount, botSessionScaleLocked
    global botSessionScaleMin, botSessionScaleMax
    global gameWindowID, gameWindowTitle, gameProcess

    if (botSessionActive)
        BotSessionStop("restarted")

    if (mobName = "")
        mobName := MobTemplateFolderName()

    runNumber := sessionBotRunCount + 1
    botSessionId := sessionLogId . "_bot_" . runNumber . "_" . A_TickCount
    botSessionDir := SessionLogEnsureDir() . "\" . botSessionId
    if (!FileExist(botSessionDir))
        FileCreateDir, %botSessionDir%
    botSessionSummaryPath := botSessionDir . "\summary.json"
    botSessionStartedTick := A_TickCount
    botSessionTargetMob := mobName
    botSessionTotalScans := 0
    botSessionEmptyScans := 0
    botSessionTargetScans := 0
    botSessionDetectFailures := 0
    botSessionAttacksIssued := 0
    botSessionResolvedEngagements := 0
    botSessionAcceptedCandidates := 0
    botSessionLastRoi := ""
    botSessionScaleMob := mobName
    botSessionScaleObservations := ""
    botSessionScaleCount := 0
    botSessionScaleLocked := false
    botSessionScaleMin := 0.0
    botSessionScaleMax := 0.0
    botSessionActive := true

    lines := "botSessionId: " . botSessionId . "`n"
    lines .= "targetMob: " . mobName . "`n"
    lines .= "startedAt: " . SessionLogTimestamp() . "`n"
    lines .= "gameWindowId: " . gameWindowID . "`n"
    lines .= "gameWindowTitle: " . gameWindowTitle . "`n"
    lines .= "gameProcess: " . gameProcess . "`n"
    lines .= "summaryPath: " . botSessionSummaryPath
    SessionLogWriteBlock("bot session start", lines)
    BotSessionWriteSummary("running")
    return botSessionId
}

BotSessionStop(reason := "stopped") {
    global botSessionActive, botSessionId, botSessionStartedTick
    global botSessionTotalScans, botSessionTargetScans, botSessionEmptyScans
    global botSessionDetectFailures, botSessionAttacksIssued, botSessionAcceptedCandidates

    if (!botSessionActive)
        return

    elapsedSec := (A_TickCount - botSessionStartedTick) // 1000
    lines := "botSessionId: " . botSessionId . "`n"
    lines .= "reason: " . reason . "`n"
    lines .= "endedAt: " . SessionLogTimestamp() . "`n"
    lines .= "durationSec: " . elapsedSec . "`n"
    lines .= "scans: " . botSessionTotalScans . "`n"
    lines .= "targetScans: " . botSessionTargetScans . "`n"
    lines .= "emptyScans: " . botSessionEmptyScans . "`n"
    lines .= "detectFailures: " . botSessionDetectFailures . "`n"
    lines .= "attacksIssued: " . botSessionAttacksIssued . "`n"
    lines .= "acceptedCandidates: " . botSessionAcceptedCandidates
    SessionLogWriteBlock("bot session end", lines)
    BotSessionWriteSummary(reason)
    botSessionActive := false
}

BotSessionGetId() {
    global botSessionActive, botSessionId
    return botSessionActive ? botSessionId : ""
}

BotSessionRecordScan(mobName, xs, ys, ws, hs) {
    global botSessionActive, botSessionTotalScans, botSessionLastRoi
    if (!botSessionActive)
        return
    botSessionTotalScans++
    botSessionLastRoi := xs . "," . ys . "," . ws . "," . hs
}

BotSessionRecordAttack(x, y, confidence := 0) {
    global botSessionActive, botSessionAttacksIssued
    if (!botSessionActive)
        return
    botSessionAttacksIssued++
    SessionLogWrite("DEBUG", "session", "attack #" . botSessionAttacksIssued . " @" . x . "," . y . " conf=" . confidence)
    BotSessionWriteSummary("running")
}

BotSessionRecordDetection(jsonText, elapsedMs) {
    global botSessionActive, botSessionDetectFailures, botSessionAcceptedCandidates
    global botSessionTargetScans, botSessionEmptyScans
    if (!botSessionActive)
        return

    if (jsonText = "" || !InStr(jsonText, """ok"":true")) {
        botSessionDetectFailures++
        BotSessionWriteSummary("running")
        return
    }

    accepted := 0
    pos := 1
    while (pos := RegExMatch(jsonText, "i)\{[^{}]+\}", block, pos)) {
        if (InStr(block, """accepted"":true")) {
            accepted++
            BotSessionMaybeRecordScale(block)
        }
        pos += StrLen(block)
    }

    botSessionAcceptedCandidates += accepted
    if (accepted > 0)
        botSessionTargetScans++
    else
        botSessionEmptyScans++
    BotSessionWriteSummary("running")
}

BotSessionMaybeRecordScale(block) {
    global botSessionScaleCount, botSessionScaleObservations, botSessionScaleLocked
    global botSessionScaleMin, botSessionScaleMax

    if (botSessionScaleLocked)
        return
    if (!RegExMatch(block, "i)""candidateScale"":([0-9.]+)", scaleMatch))
        return

    scale := scaleMatch1 + 0.0
    finalScore := 0.0
    purity := 0.0
    body := 0.0
    accent := 0.0
    pattern := 0.0
    rare := 0.0
    if (RegExMatch(block, "i)""finalScore"":([0-9.]+)", m))
        finalScore := m1 + 0.0
    if (RegExMatch(block, "i)""colorPurityScore"":([0-9.]+)", m))
        purity := m1 + 0.0
    if (RegExMatch(block, "i)""bodyPaletteScore"":([0-9.]+)", m))
        body := m1 + 0.0
    if (RegExMatch(block, "i)""accentScore"":([0-9.]+)", m))
        accent := m1 + 0.0
    if (RegExMatch(block, "i)""localPatternScore"":([0-9.]+)", m))
        pattern := m1 + 0.0
    if (RegExMatch(block, "i)""rareColorScore"":([0-9.]+)", m))
        rare := m1 + 0.0

    if (finalScore < 0.46 || purity < 0.55 || body < 0.35 || accent < 0.25 || pattern < 0.20)
        return
    if (rare > body * 1.15)
        return

    botSessionScaleCount++
    botSessionScaleObservations .= (botSessionScaleObservations = "" ? "" : ",") . Round(scale, 3)
    BotSessionRecomputeScaleRange()
}

BotSessionRecomputeScaleRange() {
    global botSessionScaleObservations, botSessionScaleCount, botSessionScaleLocked
    global botSessionScaleMin, botSessionScaleMax

    if (botSessionScaleCount <= 0)
        return

    sum := 0.0
    minScale := 999.0
    maxScale := 0.0
    Loop, Parse, botSessionScaleObservations, `,
    {
        value := A_LoopField + 0.0
        sum += value
        if (value < minScale)
            minScale := value
        if (value > maxScale)
            maxScale := value
    }
    mean := sum / botSessionScaleCount
    tolerance := 0.08
    observedSpread := (maxScale - minScale) / 2.0 + 0.04
    if (observedSpread > tolerance)
        tolerance := observedSpread
    botSessionScaleMin := Round(mean - tolerance, 3)
    botSessionScaleMax := Round(mean + tolerance, 3)
    if (botSessionScaleMin < 0.35)
        botSessionScaleMin := 0.35
    if (botSessionScaleMax > 1.20)
        botSessionScaleMax := 1.20
    if (botSessionScaleCount >= 2)
        botSessionScaleLocked := true
}

BotSessionScaleArgs(mobName) {
    global botSessionActive, botSessionScaleMob, botSessionScaleLocked
    global botSessionScaleMin, botSessionScaleMax

    if (!botSessionActive || !botSessionScaleLocked)
        return ""
    if (mobName != botSessionScaleMob)
        return ""
    return " --scale-range " . botSessionScaleMin . "," . botSessionScaleMax . " --enforce-size-gate"
}

BotSessionWriteSummary(status := "running") {
    global botSessionActive, botSessionSummaryPath, botSessionId, botSessionTargetMob
    global botSessionStartedTick, botSessionTotalScans, botSessionEmptyScans, botSessionTargetScans
    global botSessionDetectFailures, botSessionAttacksIssued, botSessionResolvedEngagements
    global botSessionAcceptedCandidates, botSessionLastRoi, botSessionScaleObservations
    global botSessionScaleCount, botSessionScaleLocked, botSessionScaleMin, botSessionScaleMax

    if (botSessionSummaryPath = "")
        return

    elapsedSec := (A_TickCount - botSessionStartedTick) // 1000
    json := "{"
    json .= """botSessionId"":""" . SessionLogJsonEscape(botSessionId) . """"
    json .= ",""status"":""" . SessionLogJsonEscape(status) . """"
    json .= ",""targetMob"":""" . SessionLogJsonEscape(botSessionTargetMob) . """"
    json .= ",""durationSec"":" . elapsedSec
    json .= ",""active"":" . (botSessionActive ? "true" : "false")
    json .= ",""lastRoi"":""" . SessionLogJsonEscape(botSessionLastRoi) . """"
    json .= ",""stats"":{""scans"":" . botSessionTotalScans . ",""emptyScans"":" . botSessionEmptyScans . ",""targetScans"":" . botSessionTargetScans . ",""detectFailures"":" . botSessionDetectFailures . ",""attacksIssued"":" . botSessionAttacksIssued . ",""resolvedEngagements"":" . botSessionResolvedEngagements . ",""acceptedCandidates"":" . botSessionAcceptedCandidates . "}"
    json .= ",""scaleCalibration"":{""locked"":" . (botSessionScaleLocked ? "true" : "false") . ",""observations"":""" . SessionLogJsonEscape(botSessionScaleObservations) . """,""count"":" . botSessionScaleCount . ",""min"":" . botSessionScaleMin . ",""max"":" . botSessionScaleMax . "}"
    json .= "}"

    FileDelete, %botSessionSummaryPath%
    FileAppend, %json%, %botSessionSummaryPath%, UTF-8
}

SessionLogHuntScan(mobName, xs, ys, ws, hs) {
    global gameWindowID
    focused := (gameWindowID && WinActive("ahk_id " . gameWindowID)) ? "yes" : "no"
    BotSessionRecordScan(mobName, xs, ys, ws, hs)
    SessionLogWrite("DEBUG", "hunt"
        , "scan mob=" . mobName . " roi=" . xs . "," . ys . " " . ws . "x" . hs . " gameFocused=" . focused)
}

SessionLogDetectResponse(jsonText, elapsedMs) {
    BotSessionRecordDetection(jsonText, elapsedMs)
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
    if (RegExMatch(jsonText, "i)""scaleCalibration"":\{[^}]*""status"":""([^""]+)""", scaleStatus))
        summary .= " scale=" . scaleStatus1
    if (RegExMatch(jsonText, "i)""candidateScale"":([0-9.]+)", candidateScale))
        summary .= " candidateScale=" . candidateScale1

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

    BotSessionStop(reason)

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
