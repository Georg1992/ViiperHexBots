#Requires AutoHotkey v1.1.33+

; HuntPolicy: attackable target selection only.
; Reads MobTracks; never mutates them or runs vision.
; No-target / mode behavior lives in HuntMode.ahk.

HuntPolicy_SelectTarget() {
    global huntTracks, HUNT_TRACK_DEBUG
    bestId := 0
    bestAttackCount := -1
    bestDistSq := -1

    for index, track in huntTracks {
        if (!MobTrack_IsAttackable(track)) {
            if (HUNT_TRACK_DEBUG && MobTrack_IsPending(track) && IsFunc("AppendLog")) {
                msLeft := track.pendingResultUntilTick - A_TickCount
                if (msLeft < 0)
                    msLeft := 0
                AppendLog("[HUNT] skip id=" . track.id . " reason=pending msLeft=" . msLeft)
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
