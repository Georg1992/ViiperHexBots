SendMode Input
CoordMode, Mouse, Screen

global etc_img := "images\etc_img.bmp"
global eqp_img := "images\eqp_img.bmp"
global use_img := "images\use_img.bmp"
global close_img := "images\close_img.bmp"
global cell1_img := "images\cell1_img.bmp"
global flywing_img := "images\wing_img.bmp"
global ok_img := "images\ok_img.bmp"
global empty_cell_img := "images\empty_cell_img.bmp"

global cellSize = 50
global wingcount := 0

; Game variables
global maxSp := 0
global currentSp := 0
global currentWeight := 0
global totalWeight := 0
global currentLocation := 0
global huntLastWarpTime := 0
global huntLastSkillTime := 0
global huntFastIdle := false
global HUNT_WATCH_INTERVAL_MS := 150
global HUNT_DISCOVERY_INTERVAL_MS := 1000
global huntServerBusy := false
global huntScanTimersActive := false
global huntDiscoveryReady := false

HuntAttackTrack(skillSC, track) {
    global SkillDelay, botRunning, botStopRequested

    if (!IsObject(track))
        return false

    remainingDelay := SkillDelay - (A_TickCount - track.lastAttackTick)
    if (remainingDelay > 0) {
        BotSleep(remainingDelay)
        if (!botRunning || botStopRequested)
            return false
    }

    if IsFunc("AppendLog")
        AppendLog("Hunt [engage]: attack track id=" . track.id . " @" . Round(track.x) . "," . Round(track.y) . " conf=" . Round(track.confidence, 2))
    MoveMouseTo(Round(track.x), Round(track.y))
    HuntSkillClick(skillSC)
    HuntTracks_MarkAttack(track.id)
    BotSessionRecordAttack(Round(track.x), Round(track.y), track.confidence)
    return true
}

HuntFilterCandidates(candidates, xs, ys, ws, hs, ByRef filteredCandidates) {
    filteredCandidates := []
    GetMobSearchPlayerIgnore(xs, ys, ws, hs, ignoreX, ignoreY, ignoreW, ignoreH)
    for index, candidate in candidates {
        if (candidate.dead) {
            filteredCandidates.Push(candidate)
            continue
        }
        if (MobPointInsideIgnore(candidate.x, candidate.y, ignoreX, ignoreY, ignoreW, ignoreH))
            continue
        filteredCandidates.Push(candidate)
    }
}

HuntStartScanTimers() {
    global huntScanTimersActive, HUNT_WATCH_INTERVAL_MS, HUNT_DISCOVERY_INTERVAL_MS, huntDiscoveryReady
    if (huntScanTimersActive)
        return
    huntDiscoveryReady := false
    SetTimer, HuntWatchTick, %HUNT_WATCH_INTERVAL_MS%
    SetTimer, HuntDiscoveryTick, %HUNT_DISCOVERY_INTERVAL_MS%
    SetTimer, HuntDiscoveryTick, -1
    huntScanTimersActive := true
}

HuntStopScanTimers() {
    global huntScanTimersActive, huntDiscoveryReady
    SetTimer, HuntWatchTick, Off
    SetTimer, HuntDiscoveryTick, Off
    huntScanTimersActive := false
    huntDiscoveryReady := false
}

HuntWatchTick() {
    global botRunning, botPaused, botStopRequested, huntServerBusy

    if (!botRunning || botPaused || botStopRequested || huntServerBusy)
        return

    GetHuntSearchRegion(xs, ys, ws, hs)
    if (!ws || !hs)
        return

    watchXs := []
    watchYs := []
    HuntTracks_CollectWatchPoints(watchXs, watchYs)
    if (!watchXs.MaxIndex())
        return

    mobName := MobTemplateFolderName()
    huntServerBusy := true
    jsonText := MobRecognitionWatchDetect(mobName, xs, ys, ws, hs, watchXs, watchYs)
    huntServerBusy := false

    if (!botRunning || botStopRequested)
        return
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return

    candidates := []
    MobRecognitionParseCandidates(jsonText, candidates)
    HuntTracks_ApplyWatch(candidates)
}

HuntDiscoveryTick() {
    global botRunning, botPaused, botStopRequested, huntServerBusy, huntDiscoveryReady, huntFastIdle

    if (!botRunning || botPaused || botStopRequested || huntServerBusy)
        return

    GetHuntSearchRegion(xs, ys, ws, hs)
    if (!ws || !hs)
        return

    HuntTracks_SetRoiCenter(xs + (ws // 2), ys + (hs // 2))

    mobName := MobTemplateFolderName()
    if IsFunc("BotSessionHuntScan") {
        fn := "BotSessionHuntScan"
        %fn%(mobName, xs, ys, ws, hs)
    }

    huntServerBusy := true
    jsonText := MobRecognitionDiscoveryDetect(mobName, xs, ys, ws, hs, false)
    huntServerBusy := false

    if (!botRunning || botStopRequested)
        return
    if (jsonText = "") {
        huntDiscoveryReady := true
        return
    }
    if (!MobJsonIsOk(jsonText)) {
        if IsFunc("AppendLog")
            AppendLog("Hunt: discovery failed — retrying")
        huntDiscoveryReady := true
        return
    }

    candidates := []
    MobRecognitionParseCandidates(jsonText, candidates)
    filteredCandidates := []
    HuntFilterCandidates(candidates, xs, ys, ws, hs, filteredCandidates)
    HuntTracks_Update(filteredCandidates)
    huntFastIdle := false
    huntDiscoveryReady := true
}

StartBot(){
    global botRunning, botPaused, botStopRequested
    if (MemoryFeaturesActive()) {
        totalWeight := ReadMemoryUInt(gameProcess, totalWeightAddress)
        UpdateGameStats()

        if (currentLocation == 0 || currentWeight == 0 || maxSp == 0) {
            MsgBox % "Failed to initialize game variables!`nCheck the client profile and memory addresses."
            return false
        }
    }

    ZoomOut()
    skillSC := GetKeySC(SkillButtonKey) + 0
    teleportSC := GetKeySC(TeleportButtonKey) + 0
    if IsFunc("AppendLog") {
        AppendLog("Bot hunt loop started")
        if (teleportSC = 0)
            AppendLog("WARNING: Teleport hotkey is not set")
        if (skillSC = 0)
            AppendLog("WARNING: Attack hotkey is not set")
    }
    while(botRunning && !botStopRequested) {
        if (!botRunning || botStopRequested)
            break
        while (botPaused && botRunning && !botStopRequested)
            BotSleep(100)
        if (!botRunning || botStopRequested)
            break
        if(SkillTimerButtonKey != ""){
            if (!SendKeyCombo(SkillTimerButtonKey))
                break
        }

        if (MemoryFeaturesActive()) {
            if(warperCoordsSet && (currentLocation == warperLocation)){ 
                if IsFunc("AppendLog")
                    AppendLog("At warper — moving to hunt map")
                MoveToTheMap(warperX, warperY)
            } 

            if (!botRunning || botStopRequested)
                break

            if(currentLocation != warperLocation){
                Hunt(skillSC, teleportSC) 
            } else if (warperCoordsSet && IsFunc("AppendLog")) {
                AppendLog("Still at warper — update location or clear warper coords to hunt")
                BotSleep(1000)
            }
        } else {
            Hunt(skillSC, teleportSC)
        }
        iterations++
    }
    BotSessionStop("loop ended")
}

Hunt(skillSC, teleportSC) {
    global botRunning, botPaused, botStopRequested
    global huntLastWarpTime, huntLastSkillTime, huntFastIdle
    global huntTrackClearScans, huntDiscoveryReady
    global CurrentTargetTrackId, HUNT_TRACK_UNREACHABLE_ATTACKS, SkillDelay

    postAttackSleepMs := 50
    emptyScanSleepMs := 25
    fastIdleScanSleepMs := 15

    SyncSearchRangeFromUI()

    if (teleportSC = 0) {
        if IsFunc("AppendLog")
            AppendLog("Hunt: Teleport key not set — bot cannot move")
        return
    }

    if (huntLastWarpTime == 0) {
        huntLastWarpTime := A_TickCount
        huntLastSkillTime := A_TickCount
    }

    HuntStartScanTimers()

    while(botRunning && !botPaused && !botStopRequested) {
        if (MemoryFeaturesActive())
            UpdateGameStats()

        if (SkillTimerButtonKey != "" && (A_TickCount - huntLastSkillTime) >= (SkillTimerInterval * 1000)) {
            if (!SendKeyCombo(SkillTimerButtonKey))
                break
            huntLastSkillTime := A_TickCount
            BotSleep(300)
        }

        if (MemoryFeaturesActive() && warperCoordsSet && SavePointButtonKey != "" && (A_TickCount - huntLastWarpTime) >= (TimeOnLocation * 1000)) {
            HuntStopScanTimers()
            WarpToSavePoint()
            huntLastWarpTime := A_TickCount
            BotSleep(1000)
            break
        }

        if (MemoryFeaturesActive() && WeightModifier >= 50 && currentWeight >= (totalWeight * WeightModifier / 100)) {
            ItemsToStorage()
            currentWeight := ReadMemoryUInt(gameProcess,currentWeightAddress)
        }

        if (MemoryFeaturesActive() && wingcount <= 0 && TakeFlyWings){
            GetFlyWings()
        }

        GetHuntSearchRegion(xs, ys, ws, hs)
        if (!ws || !hs) {
            if IsFunc("AppendLog")
                AppendLog("Hunt: invalid search region — select game window and refresh")
            BotSleep(500)
            continue
        }

        if (MemoryFeaturesActive() && DetectCaptcha && captchaEnabled && DetectCAPTCHA(xs, ys, ws, hs)) {
            HuntStopScanTimers()
            RequestBotStop("captcha detected")
            break
        }

        HuntTracks_ClearTargetIfDead()

        engaged := false
        if (CurrentTargetTrackId != "") {
            currentTrack := HuntTracks_GetTrackById(CurrentTargetTrackId)
            if (!IsObject(currentTrack) || currentTrack.state = "gone" || currentTrack.unreachable) {
                HuntTracks_Log("HUNT", "clear currentTarget id=" . CurrentTargetTrackId . " reason=lost")
                CurrentTargetTrackId := ""
            } else if (currentTrack.state = "dead") {
                HuntTracks_Log("HUNT", "clear currentTarget id=" . CurrentTargetTrackId . " reason=dead")
                CurrentTargetTrackId := ""
            } else if (!HuntTracks_IsEngageable(currentTrack)) {
                HuntTracks_Log("HUNT", "clear currentTarget id=" . CurrentTargetTrackId . " reason=stale")
                CurrentTargetTrackId := ""
            } else if (currentTrack.attackCount >= HUNT_TRACK_UNREACHABLE_ATTACKS) {
                HuntTracks_MarkUnreachable(CurrentTargetTrackId)
                HuntTracks_Log("HUNT", "clear currentTarget id=" . CurrentTargetTrackId . " reason=unreachable")
                CurrentTargetTrackId := ""
            } else {
                engaged := HuntAttackTrack(skillSC, currentTrack)
                if (engaged)
                    BotSleep(postAttackSleepMs)
            }
        }

        if (engaged) {
            huntFastIdle := false
            BotSleep(20)
            continue
        }

        newTargetId := HuntTracks_SelectTarget()
        if (newTargetId) {
            if (!botRunning || botStopRequested)
                break
            CurrentTargetTrackId := newTargetId
            newTrack := HuntTracks_GetTrackById(newTargetId)
            if IsFunc("AppendLog")
                AppendLog("Hunt [target]: attack track id=" . newTargetId . " @" . Round(newTrack.x) . "," . Round(newTrack.y) . " conf=" . Round(newTrack.confidence, 2) . " alive=" . HuntTracks_GetAliveCount())
            MoveMouseTo(Round(newTrack.x), Round(newTrack.y))
            HuntSkillClick(skillSC)
            HuntTracks_MarkAttack(newTargetId)
            BotSessionRecordAttack(Round(newTrack.x), Round(newTrack.y), newTrack.confidence)
            huntFastIdle := false
            BotSleep(postAttackSleepMs)
            continue
        }

        scanSleepMs := huntFastIdle ? fastIdleScanSleepMs : emptyScanSleepMs
        clearScansRequired := (huntFastIdle || HuntTracks_AllTracksDead()) ? 1 : 2

        if (huntDiscoveryReady && CurrentTargetTrackId = "" && HuntTracks_AllCleared(clearScansRequired)) {
            huntDiscoveryReady := false
            if (!botRunning || botStopRequested)
                break
            if IsFunc("AppendLog")
                AppendLog("Hunt: area clear — teleporting (alive=" . HuntTracks_GetAliveCount() . " areaClear=" . HuntTracks_GetAreaClearAliveCount() . ")")
            Teleport(teleportSC)
            HuntAreaReset()
            huntFastIdle := true
            BotSleep(40)
            continue
        }

        if (huntDiscoveryReady && HuntTracks_GetActionableAliveCount() > 0) {
            if IsFunc("AppendLog")
                AppendLog("Hunt [scan]: actionable tracks=" . HuntTracks_GetActionableAliveCount() . " waiting")
            huntDiscoveryReady := false
            BotSleep(scanSleepMs)
            continue
        }

        if (huntDiscoveryReady) {
            if IsFunc("AppendLog")
                AppendLog("Hunt [scan]: clear scan pending " . huntTrackClearScans . "/" . clearScansRequired)
            huntDiscoveryReady := false
        }
        BotSleep(scanSleepMs)
    }

    HuntStopScanTimers()
}

Teleport(teleportSC){
    global botRunning, botStopRequested
    if (!botRunning || botStopRequested)
        return
    Input.SendKey(teleportSC, 1)
    if (!BotSleep(50)) {
        Input.SendKey(teleportSC, 0)
        return
    }
    if (!botRunning || botStopRequested) {
        Input.SendKey(teleportSC, 0)
        return
    }
    Input.SendKey(teleportSC, 0)
    BotSleep(400)
    if(TakeFlyWings && MemoryFeaturesActive()){
        wingcount--
    }
}

MoveToTheMap(posX, posY) {
    if (BotShouldStop())
        return false
    MoveMouseTo(posX, posY)
    if (!BotSleep(500))
        return false
    Input.SendMouseButton(0, 1)
    if (!BotSleep(50)) {
        Input.SendMouseButton(0, 0)
        return false
    }
    Input.SendMouseButton(0, 0)
    if (!BotSleep(500))
        return false
    enterSC := GetKeySC("Enter") + 0
    Input.SendKey(enterSC, 1)
    if (!BotSleep(50)) {
        Input.SendKey(enterSC, 0)
        return false
    }
    Input.SendKey(enterSC, 0)
    if (!BotSleep(2000))
        return false
    UpdateGameStats()
    return true
}

WarpToSavePoint() {
    if (!SendKeyCombo(SavePointButtonKey))
        return false
    if (!BotSleep(2000))
        return false
    UpdateGameStats()
    return true
}

GetFlyWings() {
    if (BotShouldStop())
        return false
    if (!BotSleep(100))
        return false
    ManageInventoryWindow()
    if (BotShouldStop())
        return false
    MoveCursorToImage(cell1_img,0,40)
    if !SendKeyCombo(OpenStorageButtonKey) {
        return false
    }
    if (!BotSleep(800))
        return false
    if(CheckInventoryCell(flywing_img)){
        AltClicks(1)

    }
    if (!BotSleep(500))
        return false
    MoveCursorToImage(flywing_img)
    if (!BotSleep(100))
        return false
    Input.SendMouseButton(0, 1)
    if (!BotSleep(100)) {
        Input.SendMouseButton(0, 0)
        return false
    }
    MoveCursorToImage(etc_img,100,20)
    Input.SendMouseButton(0, 0)
    if (!BotSleep(200))
        return false
    send %wingsTaken%
    if (!BotSleep(200))
        return false
    enterSC := GetKeySC("Enter") + 0
    Input.SendKey(enterSC, 1)
    if (!BotSleep(50)) {
        Input.SendKey(enterSC, 0)
        return false
    }
    Input.SendKey(enterSC, 0)
    ManageInventoryWindow()
    MoveCursorToImage(close_img)
    if (!BotSleep(200))
        return false
    InputClick()
    wingcount := wingsTaken
    return BotSleep(200)
}

ManageInventoryWindow(){
    if (BotShouldStop())
        return false
    action := "open"
    if(action = "close"){
        ImageSearch, FoundX, FoundY, 0, 0, A_ScreenWidth, A_ScreenHeight, etc_img

    }
    Input.SendKey(56, 1)
    if (!BotSleep(50)) {
        Input.SendKey(56, 0)
        return false
    }
    Input.SendKey(18, 1)
    if (!BotSleep(50)) {
        Input.SendKey(18, 0)
        Input.SendKey(56, 0)
        return false
    }
    Input.SendKey(18, 0)
    if (!BotSleep(50)) {
        Input.SendKey(56, 0)
        return false
    }
    Input.SendKey(56, 0)
    return BotSleep(500)
}

DetectCAPTCHA(xs, ys, ws, hs) {
    global captchaColor
    PixelSearch, x, y, xs, ys, xs + ws, ys + hs, %captchaColor%, 1, Fast RGB 
    if (ErrorLevel = 0) {
        Loop,8{
            if (BotShouldStop())
                break
            SoundBeep, 750, 1000
            if (!BotSleep(500))
                break
        }
        return true
    }
    return false
}

CheckInventoryCell(image, ignoreWing := true) {
    MouseGetPos, currentX, currentY

    cellSize := 40
    searchLeft := currentX - cellSize//2
    searchTop := currentY - cellSize//2
    searchRight := currentX + cellSize//2
    searchBottom := currentY + cellSize//2

    ImageSearch, FoundX, FoundY, searchLeft, searchTop, searchRight, searchBottom, %image%

    if (ErrorLevel = 0) {
        if(image == flywing_img && ignoreWing == false){
            nextCellX := currentX + cellSize
            nextCellY := currentY

            maxRight := A_ScreenWidth - cellSize//2
            if (nextCellX > maxRight) {
                nextCellX := cellSize//2
                nextCellY += cellSize
            }
            MoveMouseTo(nextCellX, nextCellY)
        }
        return true
    }

    return false
}

ItemsToStorage(){
    if (BotShouldStop())
        return false
    if (!BotSleep(500))
        return false
    ManageInventoryWindow()
    if (!BotSleep(500))
        return false
    MoveCursorToImage(use_img)
    if (!BotSleep(100))
        return false
    InputClick()
    SendKeyCombo(OpenStorageButtonKey)
    MoveCursorToImage(cell1_img,0,40)
    while(!BotShouldStop() && !CheckInventoryCell(empty_cell_img)){
        CheckInventoryCell(flywing_img, false)
        AltClicks(1)
        if (!BotSleep(50))
            return false
    }
    if (BotShouldStop() || !BotSleep(100))
        return false
    MoveCursorToImage(eqp_img)
    if (!BotSleep(100))
        return false
    InputClick()
    if (!BotSleep(50))
        return false
    MoveCursorToImage(cell1_img,0,40)
    while(!BotShouldStop() && !CheckInventoryCell(empty_cell_img)){
        AltClicks(1)
        if (!BotSleep(50))
            return false
    }

    MoveCursorToImage(etc_img)
    if (!BotSleep(100))
        return false
    InputClick()
    if (!BotSleep(100))
        return false
    MoveCursorToImage(cell1_img,0,40)
    while(!BotShouldStop() && !CheckInventoryCell(empty_cell_img)){
        if (!BotSleep(50))
            return false
        if(CheckImageOnScreen(ok_img)){
            Input.SendKey(284, 1)
            if (!BotSleep(50)) {
                Input.SendKey(284, 0)
                return false
            }
            Input.SendKey(284, 0)
            MouseGetPos, currentX, currentY
            MoveMouseTo(currentX + 40, currentY)
        }
        AltClicks(1)
    }
    if (BotShouldStop() || !BotSleep(100))
        return false
    MoveCursorToImage(close_img,10,10)
    if (!BotSleep(100))
        return false
    InputClick()
    ManageInventoryWindow()
    return BotSleep(500)
}
