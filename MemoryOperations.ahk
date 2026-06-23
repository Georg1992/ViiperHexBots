global maxSpAddress := 0
global currentSpAddress := 0
global currentWeightAddress := 0
global totalWeightAddress := 0
global currentLocationAddress := 0

UpdateGameStats() {
    if (!MemoryFeaturesActive())
        return

    Critical
    maxSp := ReadMemoryUInt(gameProcess, maxSpAddress)
    currentSp := ReadMemoryUInt(gameProcess, currentSpAddress)
    currentWeight := ReadMemoryUInt(gameProcess, currentWeightAddress)
    currentLocation := ReadMemoryUInt(gameProcess, currentLocationAddress)
}

ReadMemoryUInt(processName, address) {
    if (!MemoryFeaturesActive() || !address)
        return 0

    Process, Exist, %processName%
    pid := ErrorLevel
    if (!pid)
        return 0

    hProcess := DllCall("OpenProcess", "UInt", 0x10, "Int", 0, "UInt", pid, "Ptr")
    if (!hProcess)
        return 0

    VarSetCapacity(buffer, 4, 0)
    success := DllCall("ReadProcessMemory", "Ptr", hProcess, "Ptr", address, "Ptr", &buffer, "UInt", 4, "UInt*", 0)
    DllCall("CloseHandle", "Ptr", hProcess)

    return (success) ? NumGet(&buffer, 0, "UInt") : 0
}
