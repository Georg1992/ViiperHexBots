#Requires AutoHotkey v1.1.33+

; Bot-session state: one active hunt session from Start Bot to Stop Bot.

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
global botSessionAcceptedCandidates := 0
global botSessionLastRoi := ""
global botSessionScaleMob := ""
global botSessionScaleObservations := ""
global botSessionScaleCount := 0
global botSessionScaleLocked := false
global botSessionScaleMin := 0.0
global botSessionScaleMax := 0.0

BotSessionStart(mobName) {
    global sessionBotRunCount, sessionLogId
    global botSessionActive, botSessionId, botSessionDir, botSessionSummaryPath, botSessionStartedTick
    global botSessionTargetMob, botSessionTotalScans, botSessionEmptyScans, botSessionTargetScans
    global botSessionDetectFailures, botSessionAttacksIssued, botSessionAcceptedCandidates
    global botSessionLastRoi, botSessionScaleMob, botSessionScaleObservations, botSessionScaleCount
    global botSessionScaleLocked, botSessionScaleMin, botSessionScaleMax
    global gameWindowID, gameWindowTitle, gameProcess

    if (botSessionActive)
        BotSessionStop("restarted")

    if (mobName = "")
        mobName := MobTemplateFolderName()

    runNumber := sessionBotRunCount + 1
    botSessionId := sessionLogId . "_bot_" . runNumber . "_" . A_TickCount
    botSessionDir := SessionLogActiveDir() . "\" . botSessionId
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

    if (finalScore < 0.46 || purity < 0.50 || body < 0.12 || accent < 0.18 || pattern < 0.15)
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

BotSessionScaleRangeJson(mobName) {
    global botSessionActive, botSessionScaleMob, botSessionScaleLocked
    global botSessionScaleMin, botSessionScaleMax

    if (!botSessionActive || !botSessionScaleLocked)
        return ""
    if (mobName != botSessionScaleMob)
        return ""
    return ",""scaleRange"":[" . botSessionScaleMin . "," . botSessionScaleMax . "],""enforceSizeGate"":true"
}

BotSessionWriteSummary(status := "running") {
    global botSessionActive, botSessionSummaryPath, botSessionId, botSessionTargetMob
    global botSessionStartedTick, botSessionTotalScans, botSessionEmptyScans, botSessionTargetScans
    global botSessionDetectFailures, botSessionAttacksIssued, botSessionAcceptedCandidates
    global botSessionLastRoi, botSessionScaleObservations, botSessionScaleCount
    global botSessionScaleLocked, botSessionScaleMin, botSessionScaleMax

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
    json .= ",""stats"":{""scans"":" . botSessionTotalScans . ",""emptyScans"":" . botSessionEmptyScans . ",""targetScans"":" . botSessionTargetScans . ",""detectFailures"":" . botSessionDetectFailures . ",""attacksIssued"":" . botSessionAttacksIssued . ",""acceptedCandidates"":" . botSessionAcceptedCandidates . "}"
    json .= ",""scaleCalibration"":{""locked"":" . (botSessionScaleLocked ? "true" : "false") . ",""observations"":""" . SessionLogJsonEscape(botSessionScaleObservations) . """,""count"":" . botSessionScaleCount . ",""min"":" . botSessionScaleMin . ",""max"":" . botSessionScaleMax . "}"
    json .= "}"

    FileDelete, %botSessionSummaryPath%
    FileAppend, %json%, %botSessionSummaryPath%, UTF-8
}

BotSessionHuntScan(mobName, xs, ys, ws, hs) {
    global gameWindowID
    focused := (gameWindowID && WinActive("ahk_id " . gameWindowID)) ? "yes" : "no"
    BotSessionRecordScan(mobName, xs, ys, ws, hs)
    SessionLogWrite("DEBUG", "hunt"
        , "scan mob=" . mobName . " roi=" . xs . "," . ys . " " . ws . "x" . hs . " gameFocused=" . focused)
}

BotSessionDetectResponse(jsonText, elapsedMs) {
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
    section := MobRecognitionExtractCandidatesSection(jsonText)
    if (section != "") {
        pos := 1
        while (pos := RegExMatch(section, "i)\{[^{}]+\}", block, pos)) {
            if (InStr(block, """accepted"":true"))
                candidateCount++
            pos += StrLen(block)
        }
    } else if (MobRecognitionCandidatesParsed(jsonText)) {
        candidateCount := 0
    }

    bestX := ""
    bestY := ""
    bestConf := ""
    if (section != "") {
        pos := 1
        while (pos := RegExMatch(section, "i)\{[^{}]+\}", block, pos)) {
            if (!InStr(block, """centerX"":") || !InStr(block, """confidence"":"))
                continue
            candX := 0
            candY := 0
            candConf := 0
            if (MobRecognitionParseCandidateBlock(block, candX, candY, candConf)) {
                bestX := candX
                bestY := candY
                bestConf := candConf
                break
            }
            pos += StrLen(block)
        }
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
