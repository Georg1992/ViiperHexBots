#Requires AutoHotkey v1.1.33+

; Mob recognition bridge (simple descriptor heatmap CLI).

global mobRecognitionDebug := false
global mobRecognitionPython := ""
global mobRecognitionCli := "mob-recognition\cli.py"
global mobRecognitionShutdownDone := false
global mobRecognitionServerPid := 0
global mobRecognitionIpcDir := ""
global mobRecognitionServerReady := false

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

    tempFile := A_Temp . "\mob_recognition_python_exe.txt"
    FileDelete, %tempFile%
    resolveCmd := A_ComSpec . " /c py -3 -c ""import sys; open(r'" . tempFile . "', 'w', encoding='utf-8').write(sys.executable)"""
    RunWait, %resolveCmd%, , Hide UseErrorLevel
    if (ErrorLevel || !FileExist(tempFile)) {
        MobRecognitionLog("MobRecognition: failed to resolve Python executable")
        return ""
    }
    FileRead, pythonExe, *P65001 %tempFile%
    FileDelete, %tempFile%
    pythonExe := Trim(pythonExe, "`r`n `t")
    if (pythonExe = "" || !FileExist(pythonExe)) {
        MobRecognitionLog("MobRecognition: resolved Python executable is missing")
        return ""
    }

    mobRecognitionPython := pythonExe
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

MobRecognitionDiscoveryDetect(mobName, roiX, roiY, roiW, roiH, showProgress := false) {
    if (roiW <= 0 || roiH <= 0) {
        MobRecognitionLog("MobRecognition: invalid hunt ROI " . roiX . "," . roiY . " " . roiW . "x" . roiH)
        return ""
    }
    if (!MobRecognitionEnsureServer()) {
        MobRecognitionLog("MobRecognition: detector unavailable")
        return ""
    }
    requestJson := MobRecognitionBuildScanRequest(mobName, roiX, roiY, roiW, roiH)
    startTick := A_TickCount
    jsonText := MobRecognitionSendServerRequest(requestJson, 60000)
    if (showProgress && IsFunc("AppendLog") && jsonText != "")
        AppendLog("Mob detect " . Round((A_TickCount - startTick) / 1000, 2) . "s")
    if (jsonText != "" && IsFunc("BotSessionDetectResponse")) {
        fn := "BotSessionDetectResponse"
        %fn%(jsonText, A_TickCount - startTick)
    }
    return jsonText
}

MobRecognitionBuildScanRequest(mobName, roiX, roiY, roiW, roiH) {
    sessionId := ""
    if IsFunc("BotSessionGetId") {
        fn := "BotSessionGetId"
        sessionId := %fn%()
    }
    scaleJson := ""
    if IsFunc("BotSessionScaleRangeJson") {
        fn := "BotSessionScaleRangeJson"
        scaleJson := %fn%(mobName)
    }
    return "{""cmd"":""scan"",""mob"":""" . mobName . """,""roi"":[" . roiX . "," . roiY . "," . roiW . "," . roiH . "],""sessionId"":""" . sessionId . """" . scaleJson . "}"
}

MobRecognitionSendServerRequest(requestJson, timeoutMs := 60000) {
    global mobRecognitionIpcDir

    if (!MobRecognitionStartServer())
        return ""

    requestFile := mobRecognitionIpcDir . "\request.json"
    responseFile := mobRecognitionIpcDir . "\response.json"
    FileDelete, %responseFile%
    FileDelete, %requestFile%
    if (!MobRecognitionWriteUtf8File(requestFile, requestJson))
        return ""

    jsonText := MobRecognitionWaitForJsonFile(responseFile, timeoutMs)
    if (jsonText = "") {
        MobRecognitionLog("MobRecognition: server request timed out")
        MobRecognitionStopServer()
        return ""
    }
    return jsonText
}

MobRecognitionClearIpcFiles(ipcDir) {
    if (ipcDir = "")
        return
    FileDelete, %ipcDir%\ready.json
    FileDelete, %ipcDir%\request.json
    FileDelete, %ipcDir%\response.json
    FileDelete, %ipcDir%\ready.json.tmp
    FileDelete, %ipcDir%\request.json.tmp
    FileDelete, %ipcDir%\response.json.tmp
}

MobRecognitionWaitForJsonFile(filePath, timeoutMs) {
    deadline := A_TickCount + timeoutMs
    jsonText := ""
    while (A_TickCount < deadline) {
        if (FileExist(filePath)) {
            FileRead, jsonText, *P65001 %filePath%
            if (MobJsonIsComplete(jsonText))
                return jsonText
        }
        Sleep, 25
    }
    return ""
}

MobRecognitionStartServer() {
    global mobRecognitionCli, mobRecognitionServerPid, mobRecognitionIpcDir, mobRecognitionServerReady

    if (mobRecognitionServerReady && mobRecognitionServerPid && MobRecognitionProcessRunning(mobRecognitionServerPid))
        return true

    MobRecognitionStopServer()

    pythonCmd := EnsureMobRecognitionPython()
    if (pythonCmd = "")
        return false

    cliPath := A_ScriptDir . "\" . mobRecognitionCli
    if (!FileExist(cliPath)) {
        MobRecognitionLog("MobRecognition: cli.py not found")
        return false
    }

    mobRecognitionIpcDir := A_Temp . "\mob_recognition_ipc"
    FileCreateDir, %mobRecognitionIpcDir%
    MobRecognitionClearIpcFiles(mobRecognitionIpcDir)

    serverCmd := """" . pythonCmd . """ -u """ . cliPath . """ serve --ipc-dir """ . mobRecognitionIpcDir . """"
    Run, %serverCmd%, %A_ScriptDir%, Hide, pid
    mobRecognitionServerPid := pid

    readyFile := mobRecognitionIpcDir . "\ready.json"
    readyLine := MobRecognitionWaitForJsonFile(readyFile, 15000)
    if (readyLine = "" || !InStr(readyLine, """ready"":true")) {
        MobRecognitionLog("MobRecognition: detector server failed to start")
        MobRecognitionStopServer()
        return false
    }
    mobRecognitionServerReady := true
    MobRecognitionLog("MobRecognition: detector server ready")
    return true
}

MobRecognitionStopServer() {
    global mobRecognitionServerPid, mobRecognitionIpcDir, mobRecognitionServerReady

    ipcDir := mobRecognitionIpcDir
    if (mobRecognitionServerPid && MobRecognitionProcessRunning(mobRecognitionServerPid) && ipcDir != "") {
        requestFile := ipcDir . "\request.json"
        responseFile := ipcDir . "\response.json"
        FileDelete, %responseFile%
        FileDelete, %requestFile%
        MobRecognitionWriteUtf8File(requestFile, "{""cmd"":""shutdown""}")
        MobRecognitionWaitForJsonFile(responseFile, 3000)
    }
    if (mobRecognitionServerPid)
        MobRecognitionKillPid(mobRecognitionServerPid)
    MobRecognitionClearIpcFiles(ipcDir)
    mobRecognitionServerPid := 0
    mobRecognitionIpcDir := ""
    mobRecognitionServerReady := false
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

    RunWait, %ComSpec% /c taskkill /T /F /PID %pid%, , Hide UseErrorLevel
    MobRecognitionWaitForProcessExit(pid, 3000)
}

MobRecognitionKillOwnedDetectorProcesses() {
    cliPath := A_ScriptDir . "\" . mobRecognitionCli
    if (!FileExist(cliPath))
        return

    sweepCmd := A_ComSpec . " /c set ""MOB_REC_CLI=" . cliPath . """ && powershell -NoProfile -WindowStyle Hidden -Command ""Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($env:MOB_REC_CLI) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"""
    RunWait, %sweepCmd%, , Hide UseErrorLevel
}

MobRecognitionEnsureServer() {
    return MobRecognitionStartServer()
}

MobRecognitionExitCleanup() {
    global mobRecognitionShutdownDone

    if (mobRecognitionShutdownDone)
        return
    mobRecognitionShutdownDone := true

    MobRecognitionStopServer()
    MobRecognitionKillOwnedDetectorProcesses()

    if IsFunc("MobRecognitionLog")
        MobRecognitionLog("MobRecognition: simple detector cleanup complete")
}

MobRecognitionOnExit(ExitReason, ExitCode) {
    MobRecognitionExitCleanup()
}

OnExit("MobRecognitionOnExit")

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

GetMobSearchPlayerIgnore(xs, ys, ws, hs, ByRef ignoreX, ByRef ignoreY, ByRef ignoreW, ByRef ignoreH) {
    global cellSize

    ignoreW := cellSize * 2
    ignoreH := cellSize * 2
    ignoreX := xs + (ws // 2) - (ignoreW // 2)
    ignoreY := ys + (hs // 2) - (ignoreH // 2)
}

