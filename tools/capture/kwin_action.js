// Apply one window action selected by resourceClass or caption substring.
// KWIN_ACTION: activate | minimize | restore ; KWIN_MATCH: substring
// KWin scripts cannot read the environment, so this file is rewritten by the
// caller with the two constants it wants; keeping the logic here makes the
// action auditable.
const action = "activate";
const match = "PLACEHOLDER";
const wins = (typeof workspace.windowList === "function")
    ? workspace.windowList()
    : (typeof workspace.clientList === "function" ? workspace.clientList() : []);
let hits = [];
for (const w of wins) {
    if (!w) continue;
    const rc = String(w.resourceClass ?? "");
    const cap = String(w.caption ?? "");
    if (rc.indexOf(match) !== -1 || cap.indexOf(match) !== -1) hits.push(w);
}
for (const w of hits) {
    if (action === "activate") workspace.activeWindow = w;
    else if (action === "minimize") w.minimized = true;
    else if (action === "restore") w.minimized = false;
    console.log("ACTION " + action + " " + JSON.stringify({ caption: w.caption, rc: w.resourceClass }));
}
console.log("ACTION_COUNT " + hits.length);
