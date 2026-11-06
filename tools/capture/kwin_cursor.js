// Print the compositor's real cursor position once, for uinput-driven pointing.
// Loaded through org.kde.KWin /Scripting; output lands in the user journal.
const p = workspace.cursorPos;
console.log("CURSOR " + JSON.stringify({ x: p.x, y: p.y }));
