#Requires AutoHotkey v1.1.33+

class ViiperInput {
    static BridgeAddr := "127.0.0.1:47832"
    static BridgeExe := "viiper-input.exe"

    __New() {
        this.EnsureBridgeProcess()
        this.Init()
    }

    EnsureBridgeProcess() {
        if (this.HttpGet("/health") != "")
            return

        bridgePath := A_ScriptDir . "\" . ViiperInput.BridgeExe
        if (!FileExist(bridgePath)) {
            MsgBox, 16, ViiperHexBots, Could not find %bridgePath%`n`nRun build.ps1 first.
            ExitApp
        }

        Run, "%bridgePath%", %A_ScriptDir%, Hide, bridgePID
        deadline := A_TickCount + 45000
        while (A_TickCount < deadline) {
            if (this.HttpGet("/health") != "")
                return
            Sleep, 200
        }

        MsgBox, 16, ViiperHexBots, Input bridge failed to start.`n`nMake sure usbip-win2 is installed and reboot if needed.
        ExitApp
    }

    Init() {
        resp := this.HttpPost("/init", "{}")
        if (InStr(resp, "error"))
            this.FailInit(resp)
    }

    SendKeyEvent(deviceId, scanCode, state) {
        body := "{""sc"":" . scanCode . ",""state"":" . state . "}"
        resp := this.HttpPost("/key", body)
        if (InStr(resp, "error"))
            this.FailRequest("key", resp)
    }

    SendMouseButtonEvent(deviceId, button, state) {
        body := "{""button"":" . button . ",""state"":" . state . "}"
        resp := this.HttpPost("/mouse", body)
        if (InStr(resp, "error"))
            this.FailRequest("mouse", resp)
    }

    Shutdown() {
        this.HttpPost("/shutdown", "{}")
    }

    HttpGet(path) {
        url := "http://" . ViiperInput.BridgeAddr . path
        try {
            whr := ComObjCreate("WinHttp.WinHttpRequest.5.1")
            whr.Open("GET", url, false)
            whr.Send()
            if (whr.Status = 200)
                return whr.ResponseText
        }
        return ""
    }

    HttpPost(path, body) {
        url := "http://" . ViiperInput.BridgeAddr . path
        try {
            whr := ComObjCreate("WinHttp.WinHttpRequest.5.1")
            whr.Open("POST", url, false)
            whr.SetRequestHeader("Content-Type", "application/json")
            whr.Send(body)
            return whr.ResponseText
        }
        return ""
    }

    FailInit(resp) {
        MsgBox, 16, ViiperHexBots, VIIPER input init failed:`n%resp%`n`nInstall usbip-win2 and reboot if needed.
        ExitApp
    }

    FailRequest(kind, resp) {
        MsgBox, 16, ViiperHexBots, VIIPER %kind% request failed:`n%resp%
        ExitApp
    }
}

OnExit("ViiperInputShutdown")

ViiperInputShutdown(ExitReason, ExitCode) {
    global AHI
    if (IsObject(AHI) && AHI.Shutdown)
        AHI.Shutdown()
}
