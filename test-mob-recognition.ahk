#SingleInstance Force
#Include MobData.ahk
#Include utilityFunctions.ahk
#Include Lib\MobRecognition.ahk

CoordMode, Mouse, Screen
CoordMode, Pixel, Screen

global gameWindowID := 0
global gameWindowTitle := ""
global gameProcess := ""
global selectedMonsterIndex := 2

; F7 = use active window as game client
; F9 = run OpenCV mob detection test (debug images on)
; Esc = exit

TrayTip, Mob Recognition Test, F7 = set game window`nF9 = OpenCV detect`nEsc = exit, 5

F7::
    WinGet, gameWindowID, ID, A
    if (!gameWindowID) {
        MsgBox, 48, Mob Recognition Test, No active window.
        return
    }
    WinGetTitle, gameWindowTitle, ahk_id %gameWindowID%
    WinGet, gameProcess, ProcessName, ahk_id %gameWindowID%
    TrayTip, Mob Recognition Test, Game window:`n%gameWindowTitle%, 3
return

F9::
    TestMobRecognition()
return

Esc::
    ExitApp

TestMobRecognition() {
    global gameWindowID, MobNames, selectedMonsterIndex

    if (!gameWindowID || !WinExist("ahk_id " gameWindowID)) {
        MsgBox, 48, Mob Recognition, Focus the game window and press F7 first.
        return
    }

    RestoreWindow()
    Sleep, 300

    GetHuntSearchRegion(xs, ys, ws, hs)
    mobName := MobTemplateFolderName()
    jsonText := MobRecognitionDetect(mobName, xs, ys, ws, hs, true)

    if (jsonText = "") {
        MsgBox, 48, Mob Recognition, Detection failed.`nInstall Python deps:`n`npip install -r mob-recognition\requirements.txt
        return
    }

    MobRecognitionLogCandidates(jsonText)

    GetMobSearchPlayerIgnore(xs, ys, ws, hs, ignoreX, ignoreY, ignoreW, ignoreH)
    foundX := 0
    foundY := 0
    conf := 0
    if (MobRecognitionBestCandidateFiltered(jsonText, foundX, foundY, conf, ignoreX, ignoreY, ignoreW, ignoreH)) {
        msg := "Best match: " . MobNames[selectedMonsterIndex]
        msg .= "`nCenter: " . foundX . ", " . foundY
        msg .= "`nConfidence: " . conf
        msg .= "`n`nDebug output: mob-recognition\debug"
        MsgBox, 64, Mob Recognition, %msg%
        return
    }

    MsgBox, 48, Mob Recognition, No match.`n`nBuild descriptor:`npy -3 mob-recognition\cli.py build-simple-descriptor --mob %mobName%
}
