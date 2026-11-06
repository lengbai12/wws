// ei_input.c - send relative pointer motion through gamescope's EIS server.
//
// Why: XTest motion moves the X pointer, and the game (Wine) reads position
// deltas, so a parked pointer at the window edge silently eats every further
// move -- measured 2026-10-23 as a ~+-60 degree rotation limit. gamescope ships
// a libei server (socket $XDG_RUNTIME_DIR/gamescope-0-ei) and forwards EI
// relative motion as wl_relative_pointer events before any cursor clamping, so
// the delta reaches the client regardless of where the cursor is. This is the
// same path the reference project fps-vision-main uses (reis + a gamescope
// pointer-lock patch, see nix/gamescope-pointer-lock.patch).
//
// Usage:
//   ei_input --dx 400 --dy 0            # relative motion, one frame
//   ei_input --dx 400 --dy 0 --click    # motion then a left click
//   ei_input --probe                    # report whether a device was granted
//
// Build:
//   gcc -O2 -o tools/capture/ei_input tools/capture/ei_input.c \
//       -I/usr/include/libei-1.0 -lei
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <libei.h>
#include <linux/input-event-codes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static const char *default_socket(void) {
    static char path[512];
    const char *runtime = getenv("XDG_RUNTIME_DIR");
    if (!runtime) runtime = "/run/user/1000";
    snprintf(path, sizeof path, "%s/gamescope-0-ei", runtime);
    return path;
}

int main(int argc, char **argv) {
    double dx = 0, dy = 0;
    int click = 0, probe = 0, wait_ms = 400;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dx") && i + 1 < argc) dx = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dy") && i + 1 < argc) dy = atof(argv[++i]);
        else if (!strcmp(argv[i], "--wait-ms") && i + 1 < argc) wait_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--click")) click = 1;
        else if (!strcmp(argv[i], "--probe")) probe = 1;
        else path = argv[i];
    }
    if (!path) path = default_socket();

    struct ei *ei = ei_new_sender("wws-aim");
    if (!ei) {
        fprintf(stderr, "ei_new_sender failed\n");
        return 2;
    }
    if (ei_setup_backend_socket(ei, path) != 0) {
        fprintf(stderr, "ei_setup_backend_socket(%s) failed: %s\n", path, strerror(errno));
        return 2;
    }

    struct ei_device *dev = NULL;
    int granted = 0;
    const int rounds = 200;
    for (int i = 0; i < rounds && !dev; i++) {
        ei_dispatch(ei);
        struct ei_event *ev;
        while ((ev = ei_get_event(ei)) != NULL) {
            enum ei_event_type type = ei_event_get_type(ev);
            if (type == EI_EVENT_SEAT_ADDED) {
                struct ei_seat *seat = ei_event_get_seat(ev);
                ei_seat_bind_capabilities(seat, EI_DEVICE_CAP_POINTER,
                                          EI_DEVICE_CAP_BUTTON, NULL);
            } else if (type == EI_EVENT_DEVICE_ADDED) {
                dev = ei_event_get_device(ev);
                ei_device_start_emulating(dev, 0);
                granted = 1;
                ei_event_unref(ev);
                break;
            } else if (type == EI_EVENT_DISCONNECT) {
                fprintf(stderr, "EI server disconnected\n");
            }
            ei_event_unref(ev);
        }
        if (!dev) {
            struct timespec ts = {0, 10 * 1000 * 1000};
            nanosleep(&ts, NULL);
        }
    }

    if (!dev) {
        fprintf(stderr, "no EI device granted on %s\n", path);
        return 3;
    }
    if (probe) {
        printf("{\"socket\":\"%s\",\"device\":true}\n", path);
        ei_device_close(dev);
        ei_unref(ei);
        return 0;
    }

    if (dx != 0 || dy != 0) ei_device_pointer_motion(dev, dx, dy);
    if (click) {
        ei_device_button_button(dev, BTN_LEFT, 1);
        ei_device_frame(dev, now_us());
        struct timespec ts = {0, 60 * 1000 * 1000};
        nanosleep(&ts, NULL);
        ei_device_button_button(dev, BTN_LEFT, 0);
    }
    ei_device_frame(dev, now_us());
    printf("{\"dx\":%.3f,\"dy\":%.3f,\"click\":%s,\"device\":%s}\n", dx, dy,
           click ? "true" : "false", granted ? "true" : "false");

    // Give the server a moment to consume the frame before the socket closes.
    struct timespec settle = {0, (long)wait_ms * 1000 * 1000};
    nanosleep(&settle, NULL);
    ei_device_close(dev);
    ei_unref(ei);
    return 0;
}
