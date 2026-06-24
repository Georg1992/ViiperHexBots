#Requires AutoHotkey v1.1.33+

; HuntPolicy: target selection and teleport gating only.
; Does not run vision, parse JSON, or mutate track storage beyond what callers apply.

HuntPolicy_IsTrackSkillReady(track) {
    global SkillDelay
    if (!IsObject(track))
        return false
    if (track.lastAttackTick = 0)
        return true
    return ((A_TickCount - track.lastAttackTick) >= SkillDelay)
}

HuntPolicy_IsTrackAttackable(track) {
    if (!IsObject(track))
        return false
    if (HuntTracks_IsResultPending(track))
        return false
    if (!HuntPolicy_IsTrackSkillReady(track))
        return false
    return true
}

HuntPolicy_SelectTarget() {
    global huntTracks, HUNT_TRACK_DEBUG
    bestId := 0
    bestAttackCount := -1
    bestDistSq := -1

    for index, track in huntTracks {
        if (!HuntPolicy_IsTrackAttackable(track)) {
            if (HUNT_TRACK_DEBUG && HuntTracks_IsResultPending(track) && IsFunc("AppendLog")) {
                msLeft := track.pendingResultUntilTick - A_TickCount
                if (msLeft < 0)
                    msLeft := 0
                AppendLog("[HUNT] skip id=" . track.id . " reason=pendingResult msLeft=" . msLeft)
            }
            continue
        }

        if (bestId = 0 || track.attackCount < bestAttackCount) {
            bestAttackCount := track.attackCount
            bestId := track.id
            bestDistSq := HuntTracks_DistanceSqFromRoiCenter(track.x, track.y)
        } else if (track.attackCount = bestAttackCount) {
            distSq := HuntTracks_DistanceSqFromRoiCenter(track.x, track.y)
            if (distSq < bestDistSq || (distSq = bestDistSq && track.id < bestId)) {
                bestDistSq := distSq
                bestId := track.id
            }
        }
    }
    return bestId
}

HuntPolicy_ShouldTeleport(discoveryLivingCount) {
    global huntTeleportSC
    if (HuntTracks_GetAliveCount() > 0)
        return false
    if (discoveryLivingCount > 0)
        return false
    if (!huntTeleportSC)
        return false
    return true
}
