/* XComposite reads this window's backing pixmap without focusing or moving it.
 * Build: cc -O2 tools/capture/window_pixels.c -o state/window_pixels \
 *        $(pkg-config --cflags --libs x11 xcomposite xdamage xfixes)
 * Usage: window_pixels WINDOW [--wait-update]
 * --wait-update allows 500 ms for damage after a cleared baseline. Its JSON
 * evidence goes to stderr; PPM alone goes to stdout. Damage bounds drawable
 * updates only: upstream game/render/display latency remains unknown.
 * The caller must bind the window to a verified game PID before and after use.
 */
#define _POSIX_C_SOURCE 200809L
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/Xdamage.h>
#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define UPDATE_TIMEOUT_MS 500
#define UPDATE_SEMANTICS \
    "xdamage_after_cleared_baseline_before_xgetimage;upstream_render_latency_unknown"

/* Poll bounds new damage; the watchdog also bounds blocked Xlib roundtrips.
 * write/_exit keep this handler safe even when libc or Xlib was interrupted.
 */
static void capture_timeout(int signal_number) {
    (void)signal_number;
    static const char message[] = "XDamage window capture exceeded 5 second timeout\n";
    ssize_t ignored = write(STDERR_FILENO, message, sizeof(message) - 1);
    (void)ignored;
    _exit(1);
}

static int monotonic_ns(uint64_t *value) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        return 0;
    }
    *value = (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
    return 1;
}

static int wait_for_update(Display *d, Drawable drawable, uint64_t *lower_bound) {
    int event_base, error_base, major = 1, minor = 1;
    if (!XDamageQueryExtension(d, &event_base, &error_base) ||
            !XDamageQueryVersion(d, &major, &minor) || major < 1) {
        fprintf(stderr, "XDamage is unavailable for --wait-update\n");
        return 0;
    }
    Damage damage = XDamageCreate(d, drawable, XDamageReportNonEmpty);
    XSync(d, False);
    /* Creation can report the entire drawable without any new drawing. Sync
     * puts that notification in our queue before we drain it. NonEmpty damage
     * stays dirty and cannot notify again until the subtraction below.
     */
    int initial_events = XEventsQueued(d, QueuedAlready);
    XEvent event;
    for (int i = 0; i < initial_events; i++) XNextEvent(d, &event);

    int ok = 0;
    if (!monotonic_ns(lower_bound)) goto done;
    unsigned long clear_serial = NextRequest(d);
    XDamageSubtract(d, damage, None, None);
    /* This must keep events: an update after Subtract but before the sync
     * reply is valid evidence. Its lower bound predates the roundtrip.
     */
    XSync(d, False);
    uint64_t deadline = *lower_bound + UINT64_C(1000000) * UPDATE_TIMEOUT_MS;
    struct pollfd connection = {.fd = ConnectionNumber(d), .events = POLLIN};
    for (;;) {
        uint64_t now;
        if (!monotonic_ns(&now)) goto done;
        if (now >= deadline) {
            fprintf(stderr, "timed out after %d ms waiting for a new drawable update\n",
                    UPDATE_TIMEOUT_MS);
            goto done;
        }
        if (XPending(d)) {
            XNextEvent(d, &event);
            if (event.type == event_base + XDamageNotify) {
                XDamageNotifyEvent *update = (XDamageNotifyEvent *)&event;
                /* Serial filtering also rejects any pre-clear notification
                 * that arrived after the initial drain. This connection has
                 * made only a few requests, so sequence wrap is impossible.
                 */
                if (!update->send_event && update->damage == damage &&
                        update->drawable == drawable &&
                        update->level == XDamageReportNonEmpty &&
                        update->serial >= clear_serial &&
                        update->area.width && update->area.height) {
                    ok = 1;
                    goto done;
                }
            }
            continue;
        }
        int remaining_ms = (int)((deadline - now + UINT64_C(999999)) /
                                 UINT64_C(1000000));
        int ready = poll(&connection, 1, remaining_ms);
        if (ready < 0) {
            if (errno == EINTR) continue;
            perror("poll XDamage");
            goto done;
        }
        if (connection.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            fprintf(stderr, "X connection closed while waiting for a drawable update\n");
            goto done;
        }
    }
done:
    XDamageDestroy(d, damage);
    return ok;
}

static unsigned char channel(unsigned long pixel, unsigned long mask) {
    if (!mask) return 0;
    while (!(mask & 1)) { mask >>= 1; pixel >>= 1; }
    return (unsigned char)((pixel & mask) * 255 / mask);
}

int main(int argc, char **argv) {
    int wait_update = argc == 3 && strcmp(argv[2], "--wait-update") == 0;
    if (argc != 2 && !wait_update) {
        fprintf(stderr, "usage: %s WINDOW [--wait-update]\n", argv[0]);
        return 2;
    }
    if (wait_update) {
        struct sigaction action = {.sa_handler = capture_timeout};
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGALRM, &action, NULL) != 0) {
            perror("sigaction");
            return 1;
        }
        alarm(5);
    }
    Display *d = XOpenDisplay(NULL);
    if (!d) { fprintf(stderr, "cannot open X display\n"); return 1; }
    Window w = strtoul(argv[1], NULL, 0);
    XWindowAttributes a;
    if (!XGetWindowAttributes(d, w, &a) || a.map_state != IsViewable ||
            a.width <= 0 || a.height <= 0 || a.width > 8192 || a.height > 8192) {
        fprintf(stderr, "window is unmapped or has unsupported dimensions\n");
        XCloseDisplay(d); return 1;
    }
    Pixmap p = XCompositeNameWindowPixmap(d, w);
    XSync(d, False);
    uint64_t lower_bound = 0;
    if (wait_update && !wait_for_update(d, p, &lower_bound)) {
        XFreePixmap(d, p); XCloseDisplay(d); return 1;
    }
    XImage *im = XGetImage(d, p, 0, 0, a.width, a.height, AllPlanes, ZPixmap);
    if (!im) { XFreePixmap(d, p); XCloseDisplay(d); return 1; }
    unsigned char *row = malloc((size_t)im->width * 3);
    if (!row) { XDestroyImage(im); XFreePixmap(d, p); XCloseDisplay(d); return 1; }
    int ok = printf("P6\n%d %d\n255\n", im->width, im->height) > 0;
    for (int y = 0; ok && y < im->height; y++) {
        for (int x = 0; x < im->width; x++) {
            unsigned long v = XGetPixel(im, x, y);
            row[3*x] = channel(v, a.visual->red_mask);
            row[3*x+1] = channel(v, a.visual->green_mask);
            row[3*x+2] = channel(v, a.visual->blue_mask);
        }
        ok = fwrite(row, 3, im->width, stdout) == (size_t)im->width;
    }
    ok = fflush(stdout) == 0 && ok;
    if (ok && wait_update) {
        ok = fprintf(stderr, "{\"drawable_update_lower_bound_monotonic_ns\":%" PRIu64
                     ",\"drawable_update_semantics\":\"%s\"}\n",
                     lower_bound, UPDATE_SEMANTICS) > 0 && fflush(stderr) == 0;
    }
    free(row); XDestroyImage(im); XFreePixmap(d, p); XCloseDisplay(d);
    return ok ? 0 : 1;
}
