#Requires AutoHotkey v1.1.33+

; MobStateRecognition: state IPC bridge only — proposes observations, does not own tracks.

global MOB_STATE_DEBUG := false

MobState_Log(message) {
    global MOB_STATE_DEBUG
    if (!MOB_STATE_DEBUG)
        return
    if IsFunc("AppendLog")
        AppendLog("[STATE] " . message)
    if IsFunc("SessionLogWrite")
        SessionLogWrite("DEBUG", "state", message)
}

MobStateBuildRequestJson(mobName, roiX, roiY, roiW, roiH, requests) {
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

    tracksJson := "["
    count := requests.MaxIndex()
    Loop % count {
        req := requests[A_Index]
        if (A_Index > 1)
            tracksJson .= ","
        localX := req.x - roiX
        localY := req.y - roiY
        tracksJson .= "{""trackId"":" . req.id . ",""x"":" . localX . ",""y"":" . localY . "}"
    }
    tracksJson .= "]"

    return "{""cmd"":""state"",""mob"":""" . mobName . """,""roi"":[" . roiX . "," . roiY . "," . roiW . "," . roiH . "],""tracks"":" . tracksJson . ",""sessionId"":""" . sessionId . """" . scaleJson . "}"
}

MobStateBuildDirectRequestJson(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY) {
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

    localX := screenX - roiX
    localY := screenY - roiY
    tracksJson := "[{""trackId"":" . trackId . ",""x"":" . localX . ",""y"":" . localY . "}]"

    return "{""cmd"":""state"",""mode"":""direct"",""mob"":""" . mobName . """,""roi"":[" . roiX . "," . roiY . "," . roiW . "," . roiH . "],""tracks"":" . tracksJson . ",""sessionId"":""" . sessionId . """" . scaleJson . "}"
}

MobStateFindJsonArrayBounds(jsonText, key) {
    marker := """" . key . """:["
    markerPos := InStr(jsonText, marker)
    if (!markerPos)
        return {innerStart: 0, innerEnd: 0}

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
                return {innerStart: arrayStart + 1, innerEnd: index - 1}
        }
        index++
    }
    return {innerStart: 0, innerEnd: 0}
}

MobStateExtractUpdatesSection(jsonText) {
    bounds := MobStateFindJsonArrayBounds(jsonText, "trackUpdates")
    if (bounds.innerStart = 0 || bounds.innerEnd < bounds.innerStart)
        return ""
    return SubStr(jsonText, bounds.innerStart, bounds.innerEnd - bounds.innerStart + 1)
}

MobStateParseUpdateBlock(block, ByRef outUpdate) {
    outUpdate := ""

    trackId := 0
    if (!RegExMatch(block, "i)""trackId"":(\d+)", match))
        return false
    trackId := match1 + 0

    state := ""
    if (!RegExMatch(block, "i)""state"":""([^""]+)""", match))
        return false
    state := match1

    confidence := 0.0
    if (RegExMatch(block, "i)""confidence"":([0-9.]+)", match))
        confidence := match1 + 0.0

    x := 0
    y := 0
    if (RegExMatch(block, "i)""x"":(\d+)", match))
        x := match1 + 0
    if (RegExMatch(block, "i)""y"":(\d+)", match))
        y := match1 + 0

    update := {}
    update.id := trackId
    update.state := state
    update.confidence := confidence
    update.x := x
    update.y := y
    outUpdate := update
    return true
}

MobStateParseUpdates(jsonText, ByRef outUpdates) {
    outUpdates := []
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return 0

    section := MobStateExtractUpdatesSection(jsonText)
    if (section = "")
        return 0

    pos := 1
    while (pos := RegExMatch(section, "i)\{[^{}]+\}", block, pos)) {
        update := ""
        if (MobStateParseUpdateBlock(block, update))
            outUpdates.Push(update)
        pos += StrLen(block)
    }
    return outUpdates.MaxIndex() ? outUpdates.MaxIndex() : 0
}

MobStateRecognize(mobName, roiX, roiY, roiW, roiH, requests) {
    if (roiW <= 0 || roiH <= 0)
        return ""
    if (!IsObject(requests) || !requests.MaxIndex())
        return ""

    requestJson := MobStateBuildRequestJson(mobName, roiX, roiY, roiW, roiH, requests)
    MobState_Log("req tracks=" . requests.MaxIndex())

    jsonText := MobRecognitionSendServerRequest(requestJson, 3000)
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return ""

    return jsonText
}

MobStateRecognizeDirect(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY) {
    if (roiW <= 0 || roiH <= 0)
        return ""
    if (trackId = "" || trackId = 0)
        return ""

    requestJson := MobStateBuildDirectRequestJson(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY)
    jsonText := MobRecognitionSendServerRequest(requestJson, 3000)
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return ""

    return jsonText
}

MobStateFetchObservations(mobName, roiX, roiY, roiW, roiH, requests) {
    jsonText := MobStateRecognize(mobName, roiX, roiY, roiW, roiH, requests)
    if (jsonText = "")
        return false

    observations := []
    if (!MobStateParseUpdates(jsonText, observations))
        return false

    HuntTracks_ReceiveStateObservations(observations)
    return true
}

MobStateFetchDirectObservation(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY) {
    jsonText := MobStateRecognizeDirect(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY)
    if (jsonText = "")
        return false

    observations := []
    if (!MobStateParseUpdates(jsonText, observations))
        return false

    if (observations.MaxIndex() >= 1 && IsFunc("AppendLog")) {
        obs := observations[1]
        if (obs.state != "unknown")
            AppendLog("[STATE] direct id=" . trackId . " state=" . obs.state . " conf=" . Round(obs.confidence, 2))
    }

    HuntTracks_ReceiveStateObservations(observations)
    return true
}

; Legacy names used by BotLogic timers
MobStateRecognizeAndApply(mobName, roiX, roiY, roiW, roiH, requests) {
    return MobStateFetchObservations(mobName, roiX, roiY, roiW, roiH, requests)
}

MobStateRecognizeDirectAndApply(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY) {
    return MobStateFetchDirectObservation(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY)
}
