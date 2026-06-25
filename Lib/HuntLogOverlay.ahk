#Requires AutoHotkey v1.1.33+

; Click-through hunt log overlay on the right edge of the game client.

global huntLogOverlayEnabled := true
global huntLogOverlayVisible := false
global huntLogOverlayLastScanLiving := 0
global HuntOLStatus := ""
global HuntOLText := ""

HuntLogOverlay_GetClientRect(ByRef left, ByRef top, ByRef width, ByRef height) {
    global gameWindowID

    left := 0
    top := 0
    width := 0
    height := 0

    if (!gameWindowID || !WinExist("ahk_id " gameWindowID))
        return false

    WinGetPos, wx, wy, ww, wh, ahk_id %gameWindowID%
    ControlGetPos, cx, cy, cw, ch, , ahk_id %gameWindowID%
    if (cw > 0 && ch > 0) {
        left := wx + cx
        top := wy + cy
        width := cw
        height := ch
    } else {
        left := wx
        top := wy
        width := ww
        height := wh
    }
    return (width > 0 && height > 0)
}

HuntLogOverlay_IsHuntLine(message) {
    if (RegExMatch(message, "i)\[(HUNT|TRACK|DISCOVERY|STATE|DIRECT|MODE)\]"))
        return true
    if (RegExMatch(message, "i)^(Bot (started|stopped|paused|resumed)|WARNING:)"))
        return true
    return false
}

HuntLogOverlay_SetScanLiving(count) {
    global huntLogOverlayLastScanLiving
    huntLogOverlayLastScanLiving := count
    HuntLogOverlay_RefreshStatus()
}

HuntLogOverlay_RefreshStatus() {
    global huntLogOverlayEnabled, huntLogOverlayVisible, huntLogOverlayLastScanLiving
    global botSessionKillCount, botSessionTeleportCount, botSessionAttacksIssued

    if (!huntLogOverlayEnabled || !huntLogOverlayVisible)
        return

    alive := 0
    if IsFunc("HuntTracks_GetTrackCount")
        alive := HuntTracks_GetTrackCount()
    else if IsFunc("HuntTracks_GetAliveCount")
        alive := HuntTracks_GetAliveCount()

    kills := 0
    teleports := 0
    attacks := 0
    if (IsFunc("BotSessionGetId") && BotSessionGetId() != "") {
        kills := botSessionKillCount
        teleports := botSessionTeleportCount
        attacks := botSessionAttacksIssued
    }

    status := "Tracks:" . alive
        . "  Scan:" . huntLogOverlayLastScanLiving
        . "  K:" . kills
        . "  TP:" . teleports
        . "  Atk:" . attacks

    GuiControl, HuntOL:, HuntOLStatus, %status%
}

HuntLogOverlay_Reposition() {
    global huntLogOverlayVisible

    if (!huntLogOverlayVisible)
        return

    if (!HuntLogOverlay_GetClientRect(clientLeft, clientTop, clientWidth, clientHeight)) {
        HuntLogOverlay_Hide()
        return
    }

    overlayWidth := 300
    overlayHeight := clientHeight - 16
    if (overlayHeight > 420)
        overlayHeight := 420
    if (overlayHeight < 120)
        overlayHeight := 120

    overlayX := clientLeft + clientWidth - overlayWidth - 8
    overlayY := clientTop + 8

    textHeight := overlayHeight - 34
    textWidth := overlayWidth - 12
    Gui, HuntOL:Show, x%overlayX% y%overlayY% w%overlayWidth% h%overlayHeight% NA
    GuiControl, HuntOL:Move, HuntOLText, % "w" . textWidth . " h" . textHeight
}

HuntLogOverlay_Show() {
    global huntLogOverlayEnabled, huntLogOverlayVisible
    global HuntOLStatus, HuntOLText

    if (!huntLogOverlayEnabled)
        return false
    if (!HuntLogOverlay_GetClientRect(clientLeft, clientTop, clientWidth, clientHeight))
        return false

    if (!huntLogOverlayVisible) {
        Gui, HuntOL:Destroy
        Gui, HuntOL:New, +AlwaysOnTop -Caption +ToolWindow +E0x20
        Gui, HuntOL:Color, 1A1A1A
        Gui, HuntOL:Margin, 6, 6
        Gui, HuntOL:Font, s9 cFFD966, Consolas
        Gui, HuntOL:Add, Text, vHuntOLStatus, Hunt overlay
        Gui, HuntOL:Font, s8 cB8F0B8, Consolas
        Gui, HuntOL:Add, Edit, vHuntOLText ReadOnly -E0x200 -VScroll -HScroll -WantReturn w288 h360 Background1A1A1A cB8F0B8
        huntLogOverlayVisible := true
        HuntLogOverlay_Clear()
    }

    HuntLogOverlay_Reposition()
    HuntLogOverlay_RefreshStatus()
    SetTimer, HuntLogOverlay_RepositionTick, 400
    return true
}

HuntLogOverlay_Hide() {
    global huntLogOverlayVisible
    SetTimer, HuntLogOverlay_RepositionTick, Off
    Gui, HuntOL:Destroy
    huntLogOverlayVisible := false
}

HuntLogOverlay_Clear() {
    global huntLogOverlayVisible
    if (!huntLogOverlayVisible)
        return
    GuiControl, HuntOL:, HuntOLText, 
    GuiControl, HuntOL:, HuntOLStatus, 
}

HuntLogOverlay_OnLog(timestampedLine, rawMessage) {
    global huntLogOverlayEnabled, huntLogOverlayVisible, botRunning, botPaused

    if (!huntLogOverlayEnabled || botPaused || !botRunning)
        return
    if (!HuntLogOverlay_IsHuntLine(rawMessage))
        return

    static lastOverlayMessage := ""
    if (rawMessage = lastOverlayMessage)
        return
    lastOverlayMessage := rawMessage

    if (!huntLogOverlayVisible)
        HuntLogOverlay_Show()
    if (!huntLogOverlayVisible)
        return

    static overlayLines := []
    overlayLines.Push(timestampedLine)
    if (overlayLines.MaxIndex() > 24)
        overlayLines.RemoveAt(1)

    newText := ""
    for index, entry in overlayLines
        newText .= (newText = "" ? "" : "`r`n") . entry

    GuiControl, HuntOL:, HuntOLText, %newText%
    HuntLogOverlay_Reposition()
}

HuntLogOverlay_RepositionTick() {
    HuntLogOverlay_Reposition()
    HuntLogOverlay_RefreshStatus()
}
