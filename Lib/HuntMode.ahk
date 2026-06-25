#Requires AutoHotkey v1.1.33+

; HuntMode: mode-specific orchestration (no-target behavior).
; Shared hunt core (Discovery, State, HuntTracks, HuntPolicy, Attack) stays mode-agnostic.

global HUNT_MODE_TELEPORT := "teleport"
global HUNT_MODE_WALK := "walk"

global huntActiveMode := ""
global huntModeTeleportSC := 0
global huntModeDiscoveryAttemptsSinceReset := 0
global huntModeHasSuccessfulDiscoverySinceReset := false
global huntModeLastDiscoveryLivingCount := 0
global huntModeLastDiscoveryCompletedTick := 0
global huntModePendingImmediateDiscovery := false
global huntModeLastNoTargetBlockedLogTick := 0
global huntModeLastWaitKnownLogTick := 0

HuntMode_Init(mode, teleportSC := 0) {
    global huntActiveMode, huntModeTeleportSC
    huntActiveMode := mode
    huntModeTeleportSC := teleportSC
    if IsFunc("AppendLog")
        AppendLog("[MODE] active=" . mode)
}

HuntMode_GetActive() {
    global huntActiveMode
    return huntActiveMode
}

HuntMode_ValidateStartup(mode, teleportSC) {
    if (mode = HUNT_MODE_TELEPORT && !teleportSC)
        return false
    return true
}

HuntMode_BeginDiscoveryAttempt() {
    global huntModePendingImmediateDiscovery, huntModeDiscoveryAttemptsSinceReset

    if (huntModePendingImmediateDiscovery && IsFunc("AppendLog"))
        AppendLog("[DISCOVERY] immediate run")
    huntModePendingImmediateDiscovery := false
    huntModeDiscoveryAttemptsSinceReset++
}

HuntMode_NoteDiscoveryLivingCount(livingCount) {
    global huntModeHasSuccessfulDiscoverySinceReset, huntModeLastDiscoveryLivingCount
    global huntModeLastDiscoveryCompletedTick
    huntModeHasSuccessfulDiscoverySinceReset := true
    huntModeLastDiscoveryLivingCount := livingCount
    huntModeLastDiscoveryCompletedTick := A_TickCount
}

HuntMode_NoteDiscoveryScanCompleted(livingCount, addedCount := 0) {
    HuntMode_NoteDiscoveryLivingCount(livingCount)
    if IsFunc("AppendLog")
        AppendLog("[DISCOVERY] scan living=" . livingCount . " added=" . addedCount)
}

HuntMode_NoteDiscoveryScanFailed(reason := "") {
    if (reason != "" && IsFunc("AppendLog"))
        AppendLog("[DISCOVERY] scan failed reason=" . reason)
}

HuntMode_QueueImmediateDiscovery(reason := "post_teleport") {
    global huntModePendingImmediateDiscovery
    huntModePendingImmediateDiscovery := true
    if IsFunc("AppendLog")
        AppendLog("[DISCOVERY] immediate queued reason=" . reason)
}

HuntMode_QueueLivingRefreshDiscovery(reason := "living_refresh") {
    HuntMode_QueueImmediateDiscovery(reason)
}

HuntMode_TryRunImmediateDiscovery() {
    global huntModePendingImmediateDiscovery, huntServerBusy, botRunning, botPaused, botStopRequested

    if (!huntModePendingImmediateDiscovery)
        return false
    if (!botRunning || botPaused || botStopRequested)
        return false
    if (huntServerBusy)
        return false
    HuntDiscoveryTick()
    return true
}

HuntMode_OnAreaReset() {
    global huntModeDiscoveryAttemptsSinceReset, huntModeHasSuccessfulDiscoverySinceReset
    global huntModeLastDiscoveryLivingCount, huntModePendingImmediateDiscovery
    global huntModeLastNoTargetBlockedLogTick, huntModeLastWaitKnownLogTick
    global huntModeLastDiscoveryCompletedTick
    huntModeDiscoveryAttemptsSinceReset := 0
    huntModeHasSuccessfulDiscoverySinceReset := false
    huntModeLastDiscoveryLivingCount := 0
    huntModeLastDiscoveryCompletedTick := 0
    huntModePendingImmediateDiscovery := false
    huntModeLastNoTargetBlockedLogTick := 0
    huntModeLastWaitKnownLogTick := 0
}

HuntMode_LogNoTargetBlocked(reason) {
    global huntModeLastNoTargetBlockedLogTick
    if (A_TickCount - huntModeLastNoTargetBlockedLogTick < 2000)
        return
    huntModeLastNoTargetBlockedLogTick := A_TickCount
    if IsFunc("AppendLog")
        AppendLog("[MODE] no-target blocked reason=" . reason)
}

HuntMode_LogWaitKnownTracksNotAttackable(knownCount) {
    global huntModeLastWaitKnownLogTick
    if (A_TickCount - huntModeLastWaitKnownLogTick < 2000)
        return
    huntModeLastWaitKnownLogTick := A_TickCount
    if IsFunc("AppendLog")
        AppendLog("[MODE] wait reason=known_tracks_not_attackable known=" . knownCount)
}

HuntMode_CanConsiderAreaClear() {
    global huntServerBusy

    if (huntServerBusy) {
        HuntMode_LogNoTargetBlocked("server_busy")
        return false
    }
    if (IsFunc("HuntHasPendingDirectState") && HuntHasPendingDirectState()) {
        HuntMode_LogNoTargetBlocked("direct_state_pending")
        return false
    }
    return true
}

HuntMode_OnNoAttackableTargets() {
    global huntActiveMode, botRunning, botPaused, botStopRequested

    if (!botRunning || botPaused || botStopRequested)
        return false
    if (HuntTracks_HasAttackableTracks())
        return false

    knownCount := HuntTracks_GetKnownTargetCount()
    if (knownCount > 0) {
        HuntMode_LogWaitKnownTracksNotAttackable(knownCount)
        return false
    }

    if (huntActiveMode = HUNT_MODE_TELEPORT)
        return HuntMode_TeleportNoTargetHandler()

    if (huntActiveMode = HUNT_MODE_WALK) {
        if IsFunc("AppendLog")
            AppendLog("[MODE] walk stub not implemented")
        return false
    }

    return false
}

HuntMode_TeleportNoTargetHandler() {
    global huntModeTeleportSC, huntModePendingImmediateDiscovery
    global huntModeHasSuccessfulDiscoverySinceReset, huntModeLastDiscoveryLivingCount

    if (!HuntMode_CanConsiderAreaClear())
        return false
    if (HuntTracks_HasKnownTargets()) {
        HuntMode_LogNoTargetBlocked("known_tracks_remain")
        return false
    }
    if (!huntModeHasSuccessfulDiscoverySinceReset) {
        HuntMode_LogNoTargetBlocked("no_discovery_yet")
        return false
    }
    if (huntModeLastDiscoveryLivingCount > 0) {
        if (!huntModePendingImmediateDiscovery)
            HuntMode_QueueLivingRefreshDiscovery("known_zero_living_stale")
        HuntMode_LogNoTargetBlocked("living_stale")
        return false
    }

    if (!huntModeTeleportSC)
        return false

    if IsFunc("AppendLog")
        AppendLog("[MODE] teleport known=0 living=0")

    Teleport(huntModeTeleportSC)
    HuntAreaReset()
    HuntMode_QueueImmediateDiscovery("post_teleport")

    return true
}
