// Log every toplevel KWin sees, so a target can be chosen without guessing.
const wins = (typeof workspace.windowList === "function")
    ? workspace.windowList()
    : (typeof workspace.clientList === "function" ? workspace.clientList() : []);
for (const w of wins) {
    if (!w) continue;
    console.log("WINDOW " + JSON.stringify({
        caption: w.caption ?? null,
        resourceClass: w.resourceClass ?? null,
        resourceName: w.resourceName ?? null,
        pid: w.pid ?? null,
        minimized: w.minimized ?? null,
        active: workspace.activeWindow === w,
        geometry: w.frameGeometry ? [w.frameGeometry.x, w.frameGeometry.y,
                                     w.frameGeometry.width, w.frameGeometry.height] : null,
    }));
}
console.log("WINDOW_COUNT " + wins.length);
