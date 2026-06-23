#Requires AutoHotkey v1.1.33+

; Mob recognition bridge (simple descriptor heatmap CLI).

global mobRecognitionDebug := false
global mobRecognitionPython := ""
global mobRecognitionCli := "mob-recognition\cli.py"
global mobRecognitionShutdownDone := false
global mobRecognitionActiveDetectPid := 0

MobTemplateFolderName(monsterIndex := "") {
    global MobNames, selectedMonsterIndex

    if (monsterIndex = "")
        monsterIndex := selectedMonsterIndex
    name := MobNames[monsterIndex]
    StringLower, name, name
    return name
}

EnsureMobRecognitionPython() {
    global mobRecognitionPython

    if (mobRecognitionPython != "")
        return mobRecognitionPython

    for index, cmd in ["py -3", "python", "python3"] {
        RunWait, %ComSpec% /c %cmd% --version, , Hide UseErrorLevel
        if (!ErrorLevel) {
            mobRecognitionPython := cmd
            return cmd
        }
    }
    return ""
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
    if (InStr(block, """living"":true"))
        return true
    if (InStr(block, """dead"":true"))
        return false
    return InStr(block, """accepted"":true")
}

MobRecognitionIsDeadBlock(block) {
    return InStr(block, """dead"":true")
}

MobRecognitionCountLivingInRange(jsonText, ignoreX, ignoreY, ignoreW, ignoreH) {
    count := 0
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return 0

    pos := 1
    while (pos := RegExMatch(jsonText, "i)\{[^{}]+\}", block, pos)) {
        if (!MobRecognitionIsLivingBlock(block)) {
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
        if (MobPointInsideIgnore(candX, candY, ignoreX, ignoreY, ignoreW, ignoreH)) {
            pos += StrLen(block)
            continue
        }

        count++
        pos += StrLen(block)
    }

    return count
}

MobRecognitionEngagementsResolved(jsonText, attackedXs, attackedYs, radius) {
    slotCount := attackedXs.MaxIndex()
    if (!slotCount)
        return true
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return false

    livingXs := []
    livingYs := []
    deadXs := []
    deadYs := []
    pos := 1
    while (pos := RegExMatch(jsonText, "i)\{[^{}]+\}", block, pos)) {
        candX := 0
        candY := 0
        candConf := 0
        if (!MobRecognitionParseCandidateBlock(block, candX, candY, candConf)) {
            pos += StrLen(block)
            continue
        }

        if (MobRecognitionIsDeadBlock(block)) {
            deadXs.Push(candX)
            deadYs.Push(candY)
        } else if (MobRecognitionIsLivingBlock(block)) {
            livingXs.Push(candX)
            livingYs.Push(candY)
        }
        pos += StrLen(block)
    }

    Loop %slotCount% {
        slotX := attackedXs[A_Index]
        slotY := attackedYs[A_Index]
        if (MobPointNearPoints(slotX, slotY, deadXs, deadYs, radius))
            continue
        if (MobPointNearPoints(slotX, slotY, livingXs, livingYs, radius))
            return false
    }

    return true
}

MobRecognitionSelectLivingTarget(jsonText, ByRef outX, ByRef outY, ByRef outConf, ignoreX, ignoreY, ignoreW, ignoreH, unreachableXs, unreachableYs, radius) {
    outX := 0
    outY := 0
    outConf := 0
    bestConf := 0.0

    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return false

    if (!IsObject(unreachableXs))
        unreachableXs := []
    if (!IsObject(unreachableYs))
        unreachableYs := []

    pos := 1
    while (pos := RegExMatch(jsonText, "i)\{[^{}]+\}", block, pos)) {
        if (!MobRecognitionIsLivingBlock(block)) {
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
        if (MobPointInsideIgnore(candX, candY, ignoreX, ignoreY, ignoreW, ignoreH)) {
            pos += StrLen(block)
            continue
        }
        if (MobPointNearPoints(candX, candY, unreachableXs, unreachableYs, radius)) {
            pos += StrLen(block)
            continue
        }

        if (candConf > bestConf) {
            bestConf := candConf
            outX := candX
            outY := candY
            outConf := candConf
        }
        pos += StrLen(block)
    }

    return (outX > 0 && outY > 0)
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

MobRecognitionShutdownServer() {
    return
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
    MobRecognitionShutdownServer()

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

MobRecognitionHuntScan(mobName, roiX, roiY, roiW, roiH, attackedXs, attackedYs, unreachableXs, unreachableYs, emptyScans, attackCounts, fastIdle := false, showProgress := false) {
    if (roiW <= 0 || roiH <= 0) {
        MobRecognitionLog("MobRecognition: invalid hunt ROI " . roiX . "," . roiY . " " . roiW . "x" . roiH)
        return ""
    }

    if (!MobRecognitionEnsureServer()) {
        MobRecognitionLog("MobRecognition: detector unavailable")
        return ""
    }

    jsonText := MobRecognitionDetectCli(mobName, roiX, roiY, roiW, roiH, false, showProgress)
    if (jsonText = "")
        return ""
    return MobRecognitionBuildSimpleHuntResponse(jsonText, attackedXs, attackedYs, unreachableXs, unreachableYs, emptyScans, fastIdle)
}

MobRecognitionBuildSimpleHuntResponse(jsonText, attackedXs, attackedYs, unreachableXs, unreachableYs, emptyScans, fastIdle := false) {
    attackX := 0
    attackY := 0
    attackConf := 0
    targetXs := []
    targetYs := []
    targetConfs := []
    emptyXs := []
    emptyYs := []
    livingInRange := MobRecognitionCollectHuntTargets(jsonText, targetXs, targetYs, targetConfs, 0, 0, 0, 0, attackedXs, attackedYs, 72, unreachableXs, unreachableYs, 72)
    engagementsResolved := MobRecognitionEngagementsResolved(jsonText, attackedXs, attackedYs, 72)

    if (livingInRange > 0) {
        attackX := targetXs[1]
        attackY := targetYs[1]
        attackConf := targetConfs[1]
    }

    teleportScansRequired := fastIdle ? 1 : 2
    canTeleport := (livingInRange = 0 && engagementsResolved && (emptyScans + 1) >= teleportScansRequired) ? "true" : "false"
    status := (livingInRange > 0) ? "target" : (engagementsResolved ? "clear" : "wait_kill")
    attackJson := "null"
    if (attackX > 0 && attackY > 0)
        attackJson := "{""centerX"":" . attackX . ",""centerY"":" . attackY . ",""confidence"":" . attackConf . ",""accepted"":true,""living"":true,""dead"":false}"

    return "{""ok"":true,""pipeline"":""simple"",""hunt"":true,""status"":""" . status . """,""livingInRange"":" . livingInRange . ",""canTeleport"":" . canTeleport . ",""engagementsResolved"":" . (engagementsResolved ? "true" : "false") . ",""teleportScansRequired"":" . teleportScansRequired . ",""attack"":" . attackJson . ",""markUnreachable"":[]}"
}

MobRecognitionParseHuntPlan(jsonText, ByRef livingInRange, ByRef canTeleport, ByRef attackX, ByRef attackY, ByRef attackConf, ByRef huntStatus, ByRef engagementsResolved, ByRef teleportScansRequired) {
    livingInRange := 0
    canTeleport := false
    attackX := 0
    attackY := 0
    attackConf := 0
    huntStatus := ""
    engagementsResolved := true
    teleportScansRequired := 6

    if (jsonText = "" || !MobJsonIsOk(jsonText) || !InStr(jsonText, """hunt"":"))
        return false

    if (RegExMatch(jsonText, "i)""livingInRange"":(\d+)", match))
        livingInRange := match1 + 0
    if (RegExMatch(jsonText, "i)""canTeleport"":(true|false)", match))
        canTeleport := (match1 = "true")
    if (RegExMatch(jsonText, "i)""engagementsResolved"":(true|false)", match))
        engagementsResolved := (match1 = "true")
    if (RegExMatch(jsonText, "i)""teleportScansRequired"":(\d+)", match))
        teleportScansRequired := match1 + 0
    if (RegExMatch(jsonText, "i)""status"":""([^""]+)""", match))
        huntStatus := match1

    if (RegExMatch(jsonText, "i)""attack"":\{[^}]*""centerX"":(\d+)[^}]*""centerY"":(\d+)[^}]*""confidence"":([0-9.]+)", match)) {
        attackX := match1 + 0
        attackY := match2 + 0
        attackConf := match3 + 0
    }

    return true
}

MobRecognitionApplyHuntMarkUnreachable(jsonText, ByRef unreachableXs, ByRef unreachableYs, radius) {
    marked := 0
    if (jsonText = "" || !InStr(jsonText, """markUnreachable"":"))
        return 0

    pos := 1
    while (pos := RegExMatch(jsonText, "i)""markUnreachable"":\[\[(\d+),(\d+)\]", match, pos)) {
        x := match1 + 0
        y := match2 + 0
        if (MobRecognitionMarkUnreachable(x, y, unreachableXs, unreachableYs, radius)) {
            marked++
            if IsFunc("AppendLog")
                AppendLog("Hunt: slot @" . x . "," . y . " abandoned — marked unreachable")
        }
        pos += StrLen(match)
    }

    return marked
}

MobRecognitionDetectCli(mobName, roiX, roiY, roiW, roiH, debug := false, showProgress := false) {
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
    cmd := A_ComSpec . " /c " . pythonCmd . " """ . cliPath . """ detect-simple --mob " . mobName . " --roi " . roiArg . " --output """ . outFile . """" . sessionArg . scaleArg . debugFlag

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
    if IsFunc("SessionLogDetectResponse") {
        fn := "SessionLogDetectResponse"
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

MobRecognitionIsHuntEligiblePoint(x, y, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs, attackedRadius, unreachableXs, unreachableYs, unreachableRadius) {
    if (MobPointInsideIgnore(x, y, ignoreX, ignoreY, ignoreW, ignoreH))
        return false
    if (!IsObject(attackedXs))
        attackedXs := []
    if (!IsObject(attackedYs))
        attackedYs := []
    if (!IsObject(unreachableXs))
        unreachableXs := []
    if (!IsObject(unreachableYs))
        unreachableYs := []
    if (MobPointNearPoints(x, y, attackedXs, attackedYs, attackedRadius))
        return false
    if (MobPointNearPoints(x, y, unreachableXs, unreachableYs, unreachableRadius))
        return false
    return true
}

MobRecognitionRecordAttackSlot(x, y, ByRef attackedXs, ByRef attackedYs, ByRef attackCounts, radius) {
    count := attackedXs.MaxIndex()
    if (count) {
        radiusSq := radius * radius
        Loop %count% {
            dx := x - attackedXs[A_Index]
            dy := y - attackedYs[A_Index]
            if ((dx * dx) + (dy * dy) <= radiusSq) {
                attackCounts[A_Index] += 1
                return A_Index
            }
        }
    }

    attackedXs.Push(x)
    attackedYs.Push(y)
    attackCounts.Push(1)
    return attackCounts.MaxIndex()
}

MobRecognitionMarkUnreachable(x, y, ByRef unreachableXs, ByRef unreachableYs, radius) {
    if (MobPointNearPoints(x, y, unreachableXs, unreachableYs, radius))
        return false

    unreachableXs.Push(x)
    unreachableYs.Push(y)
    return true
}

MobRecognitionUpdateUnreachableFromScan(jsonText, attackedXs, attackedYs, attackCounts, ByRef unreachableXs, ByRef unreachableYs, ignoreX, ignoreY, ignoreW, ignoreH, radius, attacksBeforeUnreachable) {
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return 0

    marked := 0
    slotCount := attackedXs.MaxIndex()
    if (!slotCount)
        return 0

    huntXs := []
    huntYs := []
    huntConfs := []
    emptyXs := []
    emptyYs := []
    emptyUnreachableXs := []
    emptyUnreachableYs := []
    MobRecognitionCollectHuntTargets(jsonText, huntXs, huntYs, huntConfs, ignoreX, ignoreY, ignoreW, ignoreH, emptyXs, emptyYs, radius, emptyUnreachableXs, emptyUnreachableYs, radius)

    Loop %slotCount% {
        if (attackCounts[A_Index] < attacksBeforeUnreachable)
            continue

        slotX := attackedXs[A_Index]
        slotY := attackedYs[A_Index]
        if (!MobPointNearPoints(slotX, slotY, huntXs, huntYs, radius))
            continue

        if (MobRecognitionMarkUnreachable(slotX, slotY, unreachableXs, unreachableYs, radius)) {
            marked++
            if IsFunc("AppendLog")
                AppendLog("Hunt: mob @" . slotX . "," . slotY . " marked unreachable (attacks not landing)")
        }
    }

    return marked
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

MobRecognitionCountInSearchRange(jsonText, ignoreX, ignoreY, ignoreW, ignoreH, unreachableXs := "", unreachableYs := "", radius := 72) {
    targetXs := []
    targetYs := []
    targetConfs := []
    attackedXs := []
    attackedYs := []
    return MobRecognitionCollectHuntTargets(jsonText, targetXs, targetYs, targetConfs, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs, radius, unreachableXs, unreachableYs, radius)
}

MobRecognitionPurgeAttacked(ByRef attackedXs, ByRef attackedYs, ByRef attackedAt, maxAgeMs) {
    now := A_TickCount
    index := 1
    while (index <= attackedXs.MaxIndex()) {
        if (now - attackedAt[index] > maxAgeMs) {
            attackedXs.RemoveAt(index)
            attackedYs.RemoveAt(index)
            attackedAt.RemoveAt(index)
        } else {
            index++
        }
    }
}

MobRecognitionIsAttackSuppressed(x, y, attackedXs, attackedYs, attackedAt, suppressMs, radius) {
    count := attackedXs.MaxIndex()
    if (!count)
        return false

    now := A_TickCount
    radiusSq := radius * radius
    Loop %count% {
        if ((now - attackedAt[A_Index]) > suppressMs)
            continue
        dx := x - attackedXs[A_Index]
        dy := y - attackedYs[A_Index]
        if ((dx * dx) + (dy * dy) <= radiusSq)
            return true
    }
    return false
}

MobRecognitionSelectHuntTarget(jsonText, ByRef outX, ByRef outY, ByRef outConf, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs, attackedAt, suppressMs := 0, suppressRadius := 72, unreachableXs := "", unreachableYs := "") {
    outX := 0
    outY := 0
    outConf := 0

    targetXs := []
    targetYs := []
    targetConfs := []
    emptyXs := []
    emptyYs := []
    count := MobRecognitionCollectHuntTargets(jsonText, targetXs, targetYs, targetConfs, ignoreX, ignoreY, ignoreW, ignoreH, emptyXs, emptyYs, suppressRadius, unreachableXs, unreachableYs, suppressRadius)
    if (count = 0)
        return 0

    bestIdx := 0
    bestConf := 0.0
    Loop %count% {
        x := targetXs[A_Index]
        y := targetYs[A_Index]
        if (MobRecognitionIsAttackSuppressed(x, y, attackedXs, attackedYs, attackedAt, suppressMs, suppressRadius))
            continue
        conf := targetConfs[A_Index]
        if (conf > bestConf) {
            bestConf := conf
            bestIdx := A_Index
        }
    }

    if (bestIdx = 0)
        return 0

    outX := targetXs[bestIdx]
    outY := targetYs[bestIdx]
    outConf := targetConfs[bestIdx]
    return bestIdx
}

MobRecognitionCountHuntTargets(jsonText, ignoreX, ignoreY, ignoreW, ignoreH, unreachableXs := "", unreachableYs := "", radius := 72) {
    return MobRecognitionCountInSearchRange(jsonText, ignoreX, ignoreY, ignoreW, ignoreH, unreachableXs, unreachableYs, radius)
}

MobRecognitionCollectHuntTargets(jsonText, ByRef outXs, ByRef outYs, ByRef outConfs, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs, attackedRadius := 72, unreachableXs := "", unreachableYs := "", unreachableRadius := 72) {
    outXs := []
    outYs := []
    outConfs := []

    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return 0

    if (!InStr(jsonText, """best"":null") && RegExMatch(jsonText, "i)""best"":\{([^}]+)\}", bestMatch)) {
        bestBlock := "{" . bestMatch1 . "}"
        if (MobRecognitionIsHuntTargetBlock(bestBlock)) {
            candX := 0
            candY := 0
            candConf := 0
            if (MobRecognitionParseCandidateBlock(bestBlock, candX, candY, candConf)) {
                if (MobRecognitionIsHuntEligiblePoint(candX, candY, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs, attackedRadius, unreachableXs, unreachableYs, unreachableRadius)) {
                    outXs.Push(candX)
                    outYs.Push(candY)
                    outConfs.Push(candConf)
                }
            }
        }
    }

    pos := 1
    while (pos := RegExMatch(jsonText, "i)\{[^{}]+\}", block, pos)) {
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
        if (!MobRecognitionIsHuntEligiblePoint(candX, candY, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs, attackedRadius, unreachableXs, unreachableYs, unreachableRadius)) {
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

MobRecognitionCollectAccepted(jsonText, ByRef outXs, ByRef outYs, ByRef outConfs, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs, attackedRadius := 72) {
    emptyUnreachableXs := []
    emptyUnreachableYs := []
    return MobRecognitionCollectHuntTargets(jsonText, outXs, outYs, outConfs, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs, attackedRadius, emptyUnreachableXs, emptyUnreachableYs, attackedRadius)
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
    attackedXs := []
    attackedYs := []
    count := MobRecognitionCollectAccepted(jsonText, targetXs, targetYs, targetConfs, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs)
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

    pos := 1
    while (pos := RegExMatch(jsonText, "i)\{[^{}]+\}", block, pos)) {
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
        MobRecognitionLog("MobRecognition: horn @ " . candX . "," . candY . " conf=" . candConf . flags)
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
    attackedXs := []
    attackedYs := []
    count := MobRecognitionCollectAccepted(jsonText, targetXs, targetYs, targetConfs, ignoreX, ignoreY, ignoreW, ignoreH, attackedXs, attackedYs)
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

