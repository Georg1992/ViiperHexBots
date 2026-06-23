#SingleInstance Force
#Include MobData.ahk
#Include utilityFunctions.ahk
#Include Lib\MobRecognition.ahk

CoordMode, Mouse, Screen
CoordMode, Pixel, Screen

global gameWindowID := 0
global selectedMonsterIndex := 2

; F9 = run OpenCV mob detection test (debug images on)
; Esc = exit

TrayTip, Mob Recognition Test, F9 = OpenCV detect`nEsc = exit, 5

F9::
    TestMobRecognition()
return

Esc::
    ExitApp
