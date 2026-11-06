// uinput_input.c - drive the real desktop seat through a uinput virtual mouse.
//
// Why: a gamescope session in windowed (Wayland) mode maps the *seat* pointer into
// its Xwayland; XTest motion inside the session does not move the compositor's
// pointer, so absolute clicks in a DirectX client land wherever the seat pointer
// happens to be.  A uinput device is a real kernel input device, so the pointer
// moves for every client (and gamescope forwards it like a physical mouse).
//
// Usage:
//   uinput_input --dx -10000 --dy -10000          # saturate to the top-left corner
//   uinput_input --dx 540 --dy 1410 --click        # then move and click
//   uinput_input --dx 0 --dy 0 --click --hold-ms 120
//   uinput_input --wheel 3
//
// Relative units are device units; the compositor applies its own acceleration,
// so calibrate by saturating a corner first and observing the hover state.
//
// Build: gcc -O2 -o tools/capture/uinput_input tools/capture/uinput_input.c
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/input-event-codes.h>
#include <linux/uinput.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int fd = -1;

static void emit(int type, int code, int value) {
    struct input_event ev = {.type = type, .code = code, .value = value};
    if (write(fd, &ev, sizeof ev) != sizeof ev) {
        fprintf(stderr, "write: %s\n", strerror(errno));
    }
}

static void sync_now(void) { emit(EV_SYN, SYN_REPORT, 0); }

static void sleep_ms(int ms) {
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

static int setup(void) {
    fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0) {
        fprintf(stderr, "open /dev/uinput: %s\n", strerror(errno));
        return -1;
    }
    ioctl(fd, UI_SET_EVBIT, EV_KEY);
    ioctl(fd, UI_SET_EVBIT, EV_REL);
    ioctl(fd, UI_SET_EVBIT, EV_SYN);
    ioctl(fd, UI_SET_RELBIT, REL_X);
    ioctl(fd, UI_SET_RELBIT, REL_Y);
    ioctl(fd, UI_SET_RELBIT, REL_WHEEL);
    ioctl(fd, UI_SET_KEYBIT, BTN_LEFT);
    ioctl(fd, UI_SET_KEYBIT, BTN_RIGHT);
    ioctl(fd, UI_SET_KEYBIT, KEY_ESC);
    ioctl(fd, UI_SET_KEYBIT, KEY_ENTER);
    ioctl(fd, UI_SET_KEYBIT, KEY_TAB);
    ioctl(fd, UI_SET_KEYBIT, KEY_SPACE);
    ioctl(fd, UI_SET_KEYBIT, KEY_LEFTSHIFT);
    for (int k = KEY_1; k <= KEY_8; k++) ioctl(fd, UI_SET_KEYBIT, k);
    // Linux letter keycodes follow physical keyboard rows, not alphabet order.
    static const int letters[] = {
        KEY_A, KEY_B, KEY_C, KEY_D, KEY_E, KEY_F, KEY_G, KEY_H, KEY_I,
        KEY_J, KEY_K, KEY_L, KEY_M, KEY_N, KEY_O, KEY_P, KEY_Q, KEY_R,
        KEY_S, KEY_T, KEY_U, KEY_V, KEY_W, KEY_X, KEY_Y, KEY_Z
    };
    for (size_t i = 0; i < sizeof letters / sizeof letters[0]; i++)
        ioctl(fd, UI_SET_KEYBIT, letters[i]);

    struct uinput_setup usetup;
    memset(&usetup, 0, sizeof usetup);
    usetup.id.bustype = BUS_USB;
    usetup.id.vendor = 0x1209;
    usetup.id.product = 0x0e1;
    usetup.id.version = 1;
    snprintf(usetup.name, sizeof usetup.name, "omp virtual mouse");
    if (ioctl(fd, UI_DEV_SETUP, &usetup) < 0) {
        fprintf(stderr, "UI_DEV_SETUP: %s\n", strerror(errno));
        return -1;
    }
    if (ioctl(fd, UI_DEV_CREATE) < 0) {
        fprintf(stderr, "UI_DEV_CREATE: %s\n", strerror(errno));
        return -1;
    }
    sleep_ms(250); // let the compositor pick the new device up
    return 0;
}

static void teardown(void) {
    if (fd >= 0) {
        ioctl(fd, UI_DEV_DESTROY);
        close(fd);
        fd = -1;
    }
}

int main(int argc, char **argv) {
    int dx = 0, dy = 0, wheel = 0, hold_ms = 60, chunk = 60;
    bool click = false, rclick = false;
    int keycode = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dx") && i + 1 < argc)
            dx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dy") && i + 1 < argc)
            dy = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc)
            chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wheel") && i + 1 < argc)
            wheel = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hold-ms") && i + 1 < argc)
            hold_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--click"))
            click = true;
        else if (!strcmp(argv[i], "--key") && i + 1 < argc) {
            const char *k = argv[++i];
            if (!strcasecmp(k, "esc")) keycode = KEY_ESC;
            else if (!strcasecmp(k, "enter")) keycode = KEY_ENTER;
            else if (!strcasecmp(k, "tab")) keycode = KEY_TAB;
            else if (!strcasecmp(k, "space")) keycode = KEY_SPACE;
            else keycode = (int)strtoul(k, NULL, 0);
        }
        else if (!strcmp(argv[i], "--right-click"))
            rclick = true;
        else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    if (setup() != 0) {
        teardown();
        return 1;
    }

    // Motion is emitted in small chunks so the compositor sees a continuous
    // movement rather than one huge jump (which can trip acceleration).
    int remaining_x = dx, remaining_y = dy;
    while (remaining_x != 0 || remaining_y != 0) {
        int sx = remaining_x;
        int sy = remaining_y;
        int limit = chunk > 0 ? chunk : 100000;
        if (sx > limit) sx = limit;
        if (sx < -limit) sx = -limit;
        if (sy > limit) sy = limit;
        if (sy < -limit) sy = -limit;
        if (sx) { emit(EV_REL, REL_X, sx); remaining_x -= sx; }
        if (sy) { emit(EV_REL, REL_Y, sy); remaining_y -= sy; }
        sync_now();
        sleep_ms(6);
    }
    if (wheel) {
        emit(EV_REL, REL_WHEEL, wheel);
        sync_now();
    }
    if (keycode >= 0) {
        sleep_ms(30);
        emit(EV_KEY, keycode, 1);
        sync_now();
        sleep_ms(hold_ms > 0 ? hold_ms : 60);
        emit(EV_KEY, keycode, 0);
        sync_now();
    }
    if (click || rclick) {
        int btn = rclick ? BTN_RIGHT : BTN_LEFT;
        sleep_ms(40);
        emit(EV_KEY, btn, 1);
        sync_now();
        sleep_ms(hold_ms > 0 ? hold_ms : 60);
        emit(EV_KEY, btn, 0);
        sync_now();
    }
    sleep_ms(80);
    teardown();
    printf("{\"dx\":%d,\"dy\":%d,\"wheel\":%d,\"click\":%s,\"hold_ms\":%d}\n",
           dx, dy, wheel, (click || rclick) ? "true" : "false", hold_ms);
    return 0;
}
