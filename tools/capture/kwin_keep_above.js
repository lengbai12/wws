// Keep the recognition viewer above everything, including the fullscreen game.
// KWin 6.8 has no wlr-layer-shell, so a true transparent overlay is unavailable;
// a small always-on-top viewer window is the workable substitute.
function matches(client) {
    var cls = (client.resourceClass || '').toString().toLowerCase();
    var name = (client.caption || '').toString().toLowerCase();
    return cls.indexOf('overlay_viewer') >= 0 || name.indexOf('recognition viewer') >= 0;
}

function apply() {
    var list = workspace.windowList ? workspace.windowList() : workspace.clientList();
    var found = 0;
    for (var i = 0; i < list.length; i++) {
        var client = list[i];
        if (matches(client)) {
            client.keepAbove = true;
            client.skipTaskbar = true;
            found++;
        }
    }
    print('keep-above applied to ' + found + ' window(s)');
}

apply();
workspace.windowAdded.connect(function (client) {
    if (matches(client)) {
        client.keepAbove = true;
        client.skipTaskbar = true;
    }
});
