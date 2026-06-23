#Requires AutoHotkey v1.1.33+

class ViiperInput {
    static BridgeAddr := "127.0.0.1:47832"
    static BridgeExe := "viiper-input.exe"

    __New() {
        this.bridgePID := 0
        this.shutdownDone := false
        this.EnsureBridgeProcess()
        this.Init()
    }

    Log(message) {
        if (IsFunc("AppendLog"))
            AppendLog(message)
    }

    SetStatus(message) {
        if (IsFunc("SetInputStatus"))
            SetInputStatus(message)
    }

    BridgeOnline() {
        return (this.HttpGet("/health") != "")
    }

    BridgeReady() {
        resp := this.HttpGet("/health")
        if (resp = "")
            return false
        return InStr(resp, """ready"":true")
    }

    EnsureBridgeProcess() {
        if (this.BridgeReady()) {
            this.Log("Virtual keyboard and mouse already ready")
            this.SetStatus("Input: Ready")
            return
        }

        if (this.BridgeOnline()) {
            this.Log("Input bridge server connected")
            return
        }

        bridgePath := A_ScriptDir . "\" . ViiperInput.BridgeExe
        if (!FileExist(bridgePath)) {
            this.Log("ERROR: viiper-input.exe not found")
            MsgBox, 16, ViiperHexBots, Could not find %bridgePath%`n`nRun build.ps1 first.
            ExitApp
        }

        this.Log("Launching viiper-input.exe...")
        this.SetStatus("Input: Launching bridge...")
        Run, "%bridgePath%", %A_ScriptDir%, Hide, bridgePID
        this.bridgePID := bridgePID

        deadline := A_TickCount + 45000
        lastStatusLog := 0
        while (A_TickCount < deadline) {
            if (this.BridgeOnline()) {
                this.Log("Input bridge server connected")
                return
            }
            if (A_TickCount - lastStatusLog > 3000) {
                this.Log("Waiting for input bridge server...")
                lastStatusLog := A_TickCount
            }
            Sleep, 200
        }

        this.Log("ERROR: Input bridge failed to start")
        MsgBox, 16, ViiperHexBots, Input bridge failed to start.`n`nMake sure usbip-win2 is installed and reboot if needed.
        ExitApp
    }

    Init() {
        if (this.BridgeReady()) {
            this.Log("Virtual keyboard and mouse ready")
            this.SetStatus("Input: Ready")
            return
        }

        this.Log("Creating virtual keyboard and mouse...")
        this.SetStatus("Input: Creating devices...")

        resp := this.HttpPost("/init", "{}", 60000)
        if (InStr(resp, "error")) {
            this.Log("ERROR: VIIPER init failed")
            this.FailInit(resp)
        }

        deadline := A_TickCount + 45000
        lastStatusLog := 0
        while (A_TickCount < deadline) {
            if (this.BridgeReady()) {
                this.Log("Virtual keyboard and mouse ready")
                this.SetStatus("Input: Ready")
                return
            }
            if (A_TickCount - lastStatusLog > 3000) {
                this.Log("Still creating virtual keyboard and mouse...")
                lastStatusLog := A_TickCount
            }
            Sleep, 500
        }

        this.Log("ERROR: Virtual devices not ready in time")
        MsgBox, 16, ViiperHexBots, Virtual keyboard and mouse did not become ready.`n`nInstall usbip-win2 and reboot if needed.
        ExitApp
    }

    SendKey(scanCode, state) {
        body := "{""sc"":" . scanCode . ",""state"":" . state . "}"
        resp := this.HttpPost("/key", body)
        if (InStr(resp, "error"))
            this.FailRequest("key", resp)
    }

    SendMouseButton(button, state) {
        body := "{""button"":" . button . ",""state"":" . state . "}"
        resp := this.HttpPost("/mouse", body)
        if (InStr(resp, "error"))
            this.FailRequest("mouse", resp)
    }

    Shutdown() {
        if (this.shutdownDone)
            return
        this.shutdownDone := true

        this.Log("Stopping virtual keyboard and mouse...")

        if (this.BridgeOnline())
            this.HttpPost("/shutdown", "{}", 10000)

        this.WaitForBridgeExit()
        this.EnsureViiperStopped()
        this.Log("VIIPER stopped")
    }

    WaitForBridgeExit() {
        bridgePID := this.bridgePID
        if (bridgePID) {
            Process, WaitClose, %bridgePID%, 5
            if (!ErrorLevel)
                return
            Process, Close, %bridgePID%
            return
        }

        Process, Exist, viiper-input.exe
        bridgePID := ErrorLevel
        if (!bridgePID)
            return

        Process, WaitClose, %bridgePID%, 5
        if (ErrorLevel)
            Process, Close, %bridgePID%
    }

    EnsureViiperStopped() {
        Process, Exist, viiper.exe
        viiperPID := ErrorLevel
        if (!viiperPID)
            return

        Process, WaitClose, %viiperPID%, 3
        if (ErrorLevel)
            Process, Close, %viiperPID%
    }

    HttpGet(path) {
        url := "http://" . ViiperInput.BridgeAddr . path
        try {
            whr := ComObjCreate("WinHttp.WinHttpRequest.5.1")
            whr.Open("GET", url, false)
            whr.SetTimeouts(5000, 5000, 5000, 5000)
            whr.Send()
            if (whr.Status = 200)
                return whr.ResponseText
        }
        return ""
    }

    HttpPost(path, body, timeoutMs := 30000) {
        url := "http://" . ViiperInput.BridgeAddr . path
        try {
            whr := ComObjCreate("WinHttp.WinHttpRequest.5.1")
            whr.Open("POST", url, false)
            whr.SetRequestHeader("Content-Type", "application/json")
            whr.SetTimeouts(timeoutMs, timeoutMs, timeoutMs, timeoutMs)
            whr.Send(body)
            return whr.ResponseText
        }
        return ""
    }

    FailInit(resp) {
        this.SetStatus("Input: Failed")
        MsgBox, 16, ViiperHexBots, VIIPER input init failed:`n%resp%`n`nInstall usbip-win2 and reboot if needed.
        ExitApp
    }

    FailRequest(kind, resp) {
        MsgBox, 16, ViiperHexBots, VIIPER %kind% request failed:`n%resp%
        ExitApp
    }
}

OnExit("ViiperInputShutdown")

ShutdownInput() {
    global Input, inputShutdownDone, viperShutdownRequested
    if (inputShutdownDone || !viperShutdownRequested)
        return
    inputShutdownDone := true

    if (IsObject(Input))
        Input.Shutdown()
}

ViiperInputShutdown(ExitReason, ExitCode) {
    if (ExitReason = "Reload")
        return

    global viperShutdownRequested
    viperShutdownRequested := true
    ShutdownInput()
}
