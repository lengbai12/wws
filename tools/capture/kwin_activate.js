// Raise and activate the game window through KWin, above Wayland terminals.
// A window-manager action: it changes stacking/focus only, never game input.
const wanted = "战舰世界";
const wins = (typeof workspace.windowList === "function")
    ? workspace.windowList()
    : (typeof workspace.clientList === "function" ? workspace.clientList() : []);
let hit = null;
for (const w of wins) {
    if (w && typeof w.caption === "string" && w.caption.indexOf(wanted) !== -1) {
        hit = w;
        break;
    }
}
if (hit) {
    workspace.activeWindow = hit;
    console.log("ACTIVATED " + hit.caption);
} else {
    console.log("NOT_FOUND " + wins.length);
}
