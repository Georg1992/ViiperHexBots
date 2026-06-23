#Requires AutoHotkey v1.1.33+

; Mob recognition bridge (simple descriptor heatmap CLI).

global mobRecognitionDebug := false
global mobRecognitionPython := ""
global mobRecognitionCli := "mob-recognition\cli.py"
global mobRecognitionShutdownDone := false
global mobRecognitionActiveDetectPid := 0

MobTemplateFolderName(monsterIndex := "") {
    global MobFolderNames, selectedMonsterIndex

    if (monsterIndex = "")
        monsterIndex := selectedMonsterIndex
    return MobFolderNames[monsterIndex]
}

EnsureMobRecognitionPython() {
    global mobRecognitionPython

    if (mobRecognitionPython != "")
        return mobRecognitionPython

    RunWait, %ComSpec% /c py -3 --version, , Hide UseErrorLevel
    if (ErrorLevel) {
        MobRecognitionLog("MobRecognition: Python 3 not found (py -3)")
        return ""
    }

    mobRecognitionPython := "py -3"
    return mobRecognitionPython
}

MobJsonIsOk(jsonText) {
    return InStr(jsonText, """ok"":true")
}

MobJsonIsComplete(jsonText) {
    return InStr(jsonText, """ok"":true") || InStr(jsonText, """ok"":false")
}

MobRecognitionLog(message) {
    if IsFunc("SessionLogWrite") {
        fn := "SessionLogWrite"
        %fn%("DEBUG", "mob", message)
    }
    if IsFunc("AppendLog") {
        fn := "AppendLog"
        %fn%(message)
    }
}

MobRecognitionShowDetectProgress(elapsed) {
    if IsFunc("ShowMobSearchHint") {
        fn := "ShowMobSearchHint"
        %fn%("Searching... " . elapsed . "s", 0, "search")
    }
}

MobRecognitionWriteUtf8File(path, text) {
    FileDelete, %path%
    file := FileOpen(path, "w", "UTF-8-RAW")
    if (!file)
        return false
    file.Write(text)
    file.Close()
    return true
}

MobRecognitionParseCandidateBlock(block, ByRef candX, ByRef candY, ByRef candConf) {
    candX := 0
    candY := 0
    candConf := 0

    if (RegExMatch(block, "i)""centerX"":(\d+)", match))
        candX := match1 + 0
    if (RegExMatch(block, "i)""centerY"":(\d+)", match))
        candY := match1 + 0
    if (RegExMatch(block, "i)""confidence"":([0-9.]+)", match))
        candConf := match1 + 0

    return (candX > 0 && candY > 0 && candConf > 0)
}

MobRecognitionParseCandidateFlags(block, ByRef dead, ByRef unreachable) {
    dead := InStr(block, """dead"":true") ? true : false
    unreachable := InStr(block, """unreachable"":true") ? true : false
}

MobRecognitionIsHuntTargetBlock(block) {
    if (!InStr(block, """accepted"":true"))
        return false
    dead := false
    unreachable := false
    MobRecognitionParseCandidateFlags(block, dead, unreachable)
    return (!dead && !unreachable)
}

MobRecognitionIsLivingBlock(block) {
    if (InStr(block, """living"":true") || InStr(block, """living"": true"))
        return true
    return false
}

MobRecognitionIsDeadBlock(block) {
    return InStr(block, """dead"":true") || InStr(block, """dead"": true")
}

MobRecognitionFindJsonArrayBounds(jsonText, key) {
    marker := """" . key . """:["
    markerPos := InStr(jsonText, marker)
    if (!markerPos)
        return {start: 0, end: 0, innerStart: 0, innerEnd: 0}

    arrayStart := markerPos + StrLen(marker) - 1
    depth := 0
    index := arrayStart
    length := StrLen(jsonText)
    while (index <= length) {
        ch := SubStr(jsonText, index, 1)
        if (ch = "[")
            depth++
        else if (ch = "]") {
            depth--
            if (depth = 0)
                return {start: arrayStart, end: index, innerStart: arrayStart + 1, innerEnd: index - 1}
        }
        index++
    }
    return {start: 0, end: 0, innerStart: 0, innerEnd: 0}
}

MobRecognitionCandidatesParsed(jsonText) {
    return MobRecognitionFindJsonArrayBounds(jsonText, "candidates").start > 0
}

MobRecognitionExtractCandidatesSection(jsonText) {
    bounds := MobRecognitionFindJsonArrayBounds(jsonText, "candidates")
    if (bounds.innerStart = 0 || bounds.innerEnd < bounds.innerStart)
        return ""
    return SubStr(jsonText, bounds.innerStart, bounds.innerEnd - bounds.innerStart + 1)
}

MobRecognitionExtractCandidatesJson(jsonText) {
    bounds := MobRecognitionFindJsonArrayBounds(jsonText, "candidates")
    if (bounds.start = 0)
        return "[]"
    return SubStr(jsonText, bounds.start, bounds.end - bounds.start + 1)
}

MobRecognitionParseCandidateLivingDead(block, ByRef living, ByRef dead) {
    living := (InStr(block, """living"":true") || InStr(block, """living"": true")) ? true : false
    dead := (InStr(block, """dead"":true") || InStr(block, """dead"": true")) ? true : false
}

MobRecognitionParseCandidates(jsonText, ByRef outCandidates) {
    outCandidates := []
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return 0

    section := MobRecognitionExtractCandidatesSection(jsonText)
    if (section = "")
        return 0

    pos := 1
    while (pos := RegExMatch(section, "i)\{[^{}]+\}", block, pos)) {
        if (!MobRecognitionIsAcceptedBlock(block)) {
            pos += StrLen(block)
            continue
        }

        candX := 0
        candY := 0
        candConf := 0
        if (!MobRecognitionParseCandidateBlock(block, candX, candY, candConf)) {
            pos += StrLen(block)
            continue
        }

        living := false
        dead := false
        MobRecognitionParseCandidateLivingDead(block, living, dead)
        candidate := {}
        candidate.x := candX
        candidate.y := candY
        candidate.confidence := candConf
        candidate.living := living
        candidate.dead := dead
        outCandidates.Push(candidate)
        pos += StrLen(block)
    }

    return outCandidates.MaxIndex() ? outCandidates.MaxIndex() : 0
}

MobRecognitionHuntDetect(mobName, roiX, roiY, roiW, roiH, probeXs := "", probeYs := "", showProgress := false) {
    if (roiW <= 0 || roiH <= 0) {
        MobRecognitionLog("MobRecognition: invalid hunt ROI " . roiX . "," . roiY . " " . roiW . "x" . roiH)
        return ""
    }

    if (!MobRecognitionEnsureServer()) {
        MobRecognitionLog("MobRecognition: detector unavailable")
        return ""
    }

    if (!IsObject(probeXs))
        probeXs := []
    if (!IsObject(probeYs))
        probeYs := []
    return MobRecognitionDetectCli(mobName, roiX, roiY, roiW, roiH, false, showProgress, probeXs, probeYs)
}

MobRecognitionProcessRunning(pid) {
    if (!pid)
        return false
    Process, Exist, %pid%
    return (ErrorLevel = pid)
}

MobRecognitionWaitForProcessExit(pid, timeoutMs) {
    if (!pid)
        return true

    deadline := A_TickCount + timeoutMs
    while (A_TickCount < deadline) {
        if (!MobRecognitionProcessRunning(pid))
            return true
        Sleep, 50
    }
    return !MobRecognitionProcessRunning(pid)
}

MobRecognitionKillPid(pid) {
    if (!pid || !MobRecognitionProcessRunning(pid))
        return

    Process, Close, %pid%
    MobRecognitionWaitForProcessExit(pid, 3000)
}

MobRecognitionEnsureServer() {
    pythonCmd := EnsureMobRecognitionPython()
    if (pythonCmd = "") {
        MobRecognitionLog("MobRecognition: Python not found")
        return false
    }

    cliPath := A_ScriptDir . "\" . mobRecognitionCli
    if (!FileExist(cliPath)) {
        MobRecognitionLog("MobRecognition: cli.py not found")
        return false
    }
    return true
}

MobRecognitionCancelActiveDetect() {
    global mobRecognitionActiveDetectPid
    if (mobRecognitionActiveDetectPid) {
        MobRecognitionKillPid(mobRecognitionActiveDetectPid)
        mobRecognitionActiveDetectPid := 0
    }
}

MobRecognitionExitCleanup() {
    global mobRecognitionShutdownDone

    if (mobRecognitionShutdownDone)
        return
    mobRecognitionShutdownDone := true

    MobRecognitionCancelActiveDetect()

    if IsFunc("MobRecognitionLog")
        MobRecognitionLog("MobRecognition: simple detector cleanup complete")
}

MobRecognitionOnExit(ExitReason, ExitCode) {
    MobRecognitionExitCleanup()
}

OnExit("MobRecognitionOnExit")

MobRecognitionDetect(mobName, roiX, roiY, roiW, roiH, debug := "", showProgress := false) {
    global mobRecognitionDebug

    if (roiW <= 0 || roiH <= 0) {
        MobRecognitionLog("MobRecognition: invalid ROI " . roiX . "," . roiY . " " . roiW . "x" . roiH)
        return ""
    }

    if (!MobRecognitionEnsureServer()) {
        MobRecognitionLog("MobRecognition: detector unavailable")
        return ""
    }

    useDebug := (debug != "") ? debug : mobRecognitionDebug
    return MobRecognitionDetectCli(mobName, roiX, roiY, roiW, roiH, useDebug, showProgress)
}

MobRecognitionDetectCli(mobName, roiX, roiY, roiW, roiH, debug := false, showProgress := false, attackedXs := "", attackedYs := "") {
    global mobRecognitionCli, mobRecognitionActiveDetectPid
    global botStopRequested

    pythonCmd := EnsureMobRecognitionPython()
    if (pythonCmd = "")
        return ""

    cliPath := A_ScriptDir . "\" . mobRecognitionCli
    debugFlag := debug ? " --debug" : ""
    roiArg := roiX . "," . roiY . "," . roiW . "," . roiH
    outFile := A_Temp . "\mob_recognition_" . A_TickCount . ".json"
    sessionArg := ""
    if IsFunc("BotSessionGetId") {
        fn := "BotSessionGetId"
        activeSessionId := %fn%()
        if (activeSessionId != "")
            sessionArg := " --session-id " . activeSessionId
    }
    scaleArg := ""
    if IsFunc("BotSessionScaleArgs") {
        fn := "BotSessionScaleArgs"
        scaleArg := %fn%(mobName)
    }
    attackArg := ""
    slotCount := attackedXs.MaxIndex()
    if (slotCount > 0) {
        attackSlots := ""
        Loop %slotCount% {
            localX := attackedXs[A_Index] - roiX
            localY := attackedYs[A_Index] - roiY
            attackSlots .= localX . "," . localY . ";"
        }
        attackArg := " --attack-slots """ . attackSlots . """"
    }
    cmd := A_ComSpec . " /c " . pythonCmd . " """ . cliPath . """ detect-simple --mob " . mobName . " --roi " . roiArg . " --output """ . outFile . """" . sessionArg . scaleArg . attackArg . debugFlag

    startTick := A_TickCount
    jsonText := ""
    Run, %cmd%, %A_ScriptDir%, Hide, pid
    mobRecognitionActiveDetectPid := pid

    while (MobRecognitionProcessRunning(pid)) {
        if (botStopRequested) {
            MobRecognitionKillPid(pid)
            mobRecognitionActiveDetectPid := 0
            FileDelete, %outFile%
            return ""
        }
        elapsed := (A_TickCount - startTick) // 1000
        if (showProgress)
            MobRecognitionShowDetectProgress(elapsed)

        if (FileExist(outFile)) {
            FileRead, jsonText, *P65001 %outFile%
            if (MobJsonIsComplete(jsonText))
                break
        }

        if (elapsed >= 60) {
            Process, Close, %pid%
            MobRecognitionLog("MobRecognition: detect-simple timed out after 60s")
            mobRecognitionActiveDetectPid := 0
            FileDelete, %outFile%
            return ""
        }

        Sleep, 50
    }

    if (MobRecognitionProcessRunning(pid))
        Process, Wait, %pid%, 3
    mobRecognitionActiveDetectPid := 0

    if (!FileExist(outFile))
        return ""

    FileRead, jsonText, *P65001 %outFile%
    FileDelete, %outFile%
    if IsFunc("BotSessionDetectResponse") {
        fn := "BotSessionDetectResponse"
        %fn%(jsonText, A_TickCount - startTick)
    }
    return jsonText
}

MobPointInsideIgnore(x, y, ignoreX, ignoreY, ignoreW, ignoreH) {
    if (ignoreW <= 0 || ignoreH <= 0)
        return false
    return (x >= ignoreX && x <= ignoreX + ignoreW && y >= ignoreY && y <= ignoreY + ignoreH)
}

MobPointNearPoints(x, y, xs, ys, radius) {
    count := xs.MaxIndex()
    if (!count)
        return false
    radiusSq := radius * radius
    Loop %count% {
        dx := x - xs[A_Index]
        dy := y - ys[A_Index]
        if ((dx * dx) + (dy * dy) <= radiusSq)
            return true
    }
    return false
}

MobRecognitionIsAcceptedBlock(block) {
    return InStr(block, """accepted"":true")
}

MobRecognitionIsHuntEligiblePoint(x, y, ignoreX, ignoreY, ignoreW, ignoreH) {
    return !MobPointInsideIgnore(x, y, ignoreX, ignoreY, ignoreW, ignoreH)
}

MobRecognitionSortTargetsByConfidence(ByRef xs, ByRef ys, ByRef confs) {
    count := xs.MaxIndex()
    if (!count || count < 2)
        return

    Loop % count - 1 {
        outer := A_Index
        Loop % count - outer {
            inner := outer + A_Index
            if (confs[inner] > confs[outer]) {
                tmp := confs[outer]
                confs[outer] := confs[inner]
                confs[inner] := tmp

                tmp := xs[outer]
                xs[outer] := xs[inner]
                xs[inner] := tmp

                tmp := ys[outer]
                ys[outer] := ys[inner]
                ys[inner] := tmp
            }
        }
    }
}

MobRecognitionCollectHuntTargets(jsonText, ByRef outXs, ByRef outYs, ByRef outConfs, ignoreX, ignoreY, ignoreW, ignoreH) {
    outXs := []
    outYs := []
    outConfs := []

    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return 0

    section := MobRecognitionExtractCandidatesSection(jsonText)
    if (section = "")
        return 0

    pos := 1
    while (pos := RegExMatch(section, "i)\{[^{}]+\}", block, pos)) {
        if (!MobRecognitionIsHuntTargetBlock(block)) {
            pos += StrLen(block)
            continue
        }

        candX := 0
        candY := 0
        candConf := 0
        if (!MobRecognitionParseCandidateBlock(block, candX, candY, candConf)) {
            pos += StrLen(block)
            continue
        }
        if (!MobRecognitionIsHuntEligiblePoint(candX, candY, ignoreX, ignoreY, ignoreW, ignoreH)) {
            pos += StrLen(block)
            continue
        }

        outXs.Push(candX)
        outYs.Push(candY)
        outConfs.Push(candConf)
        pos += StrLen(block)
    }

    MobRecognitionSortTargetsByConfidence(outXs, outYs, outConfs)
    return outXs.MaxIndex() ? outXs.MaxIndex() : 0
}

MobRecognitionCollectAccepted(jsonText, ByRef outXs, ByRef outYs, ByRef outConfs, ignoreX, ignoreY, ignoreW, ignoreH) {
    return MobRecognitionCollectHuntTargets(jsonText, outXs, outYs, outConfs, ignoreX, ignoreY, ignoreW, ignoreH)
}

GetMobSearchPlayerIgnore(xs, ys, ws, hs, ByRef ignoreX, ByRef ignoreY, ByRef ignoreW, ByRef ignoreH) {
    global cellSize

    ignoreW := cellSize * 2
    ignoreH := cellSize * 2
    ignoreX := xs + (ws // 2) - (ignoreW // 2)
    ignoreY := ys + (hs // 2) - (ignoreH // 2)
}

MobRecognitionBestCandidate(jsonText, ByRef outX, ByRef outY, ByRef confidence := "") {
    return MobRecognitionBestCandidateFiltered(jsonText, outX, outY, confidence, 0, 0, 0, 0)
}

MobRecognitionBestCandidateFiltered(jsonText, ByRef outX, ByRef outY, ByRef confidence, ignoreX, ignoreY, ignoreW, ignoreH) {
    targetXs := []
    targetYs := []
    targetConfs := []
    count := MobRecognitionCollectAccepted(jsonText, targetXs, targetYs, targetConfs, ignoreX, ignoreY, ignoreW, ignoreH)
    if (count = 0) {
        outX := 0
        outY := 0
        confidence := 0
        return false
    }

    outX := targetXs[1]
    outY := targetYs[1]
    confidence := targetConfs[1]
    return true
}

MobRecognitionLogCandidates(jsonText) {
    if (!IsFunc("AppendLog") || jsonText = "")
        return

    if (!MobJsonIsOk(jsonText)) {
        MobRecognitionLog("MobRecognition: detect failed")
        return
    }

    if (InStr(jsonText, """candidates"":[]")) {
        MobRecognitionLog("MobRecognition: no candidates")
        return
    }

    section := MobRecognitionExtractCandidatesSection(jsonText)
    if (section = "")
        return

    pos := 1
    while (pos := RegExMatch(section, "i)\{[^{}]+\}", block, pos)) {
        if (!MobRecognitionIsAcceptedBlock(block)) {
            pos += StrLen(block)
            continue
        }
        candX := 0
        candY := 0
        candConf := 0
        if (!MobRecognitionParseCandidateBlock(block, candX, candY, candConf)) {
            pos += StrLen(block)
            continue
        }
        dead := false
        unreachable := false
        MobRecognitionParseCandidateFlags(block, dead, unreachable)
        flags := ""
        if (dead)
            flags .= " dead"
        if (unreachable)
            flags .= " unreachable"
        MobRecognitionLog("MobRecognition: candidate @ " . candX . "," . candY . " conf=" . candConf . flags)
        pos += StrLen(block)
    }
}

FindMobTarget(ByRef outX, ByRef outY, xs, ys, ws, hs, ignoreX := 0, ignoreY := 0, ignoreW := 0, ignoreH := 0, showProgress := false) {
    outX := 0
    outY := 0

    if (ignoreW <= 0 || ignoreH <= 0)
        GetMobSearchPlayerIgnore(xs, ys, ws, hs, ignoreX, ignoreY, ignoreW, ignoreH)

    mobName := MobTemplateFolderName()
    if (showProgress && IsFunc("AppendLog"))
        AppendLog("Mob search started (" . mobName . ")")

    jsonText := MobRecognitionDetect(mobName, xs, ys, ws, hs, "", showProgress)
    if (jsonText = "") {
        if (showProgress && IsFunc("AppendLog"))
            AppendLog("Mob search finished — detect failed")
        return false
    }

    MobRecognitionLogCandidates(jsonText)
    targetXs := []
    targetYs := []
    targetConfs := []
    count := MobRecognitionCollectAccepted(jsonText, targetXs, targetYs, targetConfs, ignoreX, ignoreY, ignoreW, ignoreH)
    if (count = 0) {
        if (showProgress && IsFunc("AppendLog"))
            AppendLog("Mob search finished — no valid match")
        return false
    }

    outX := targetXs[1]
    outY := targetYs[1]
    if (showProgress && IsFunc("AppendLog"))
        AppendLog("Mob search finished — match at " . outX . "," . outY . " conf=" . targetConfs[1])
    return true
}

