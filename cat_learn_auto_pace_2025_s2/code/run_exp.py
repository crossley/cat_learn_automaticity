# -*- coding: utf-8 -*-
"""
Single-phase categorization task (PsychoPy-only) with veridical feedback and EEG triggers.
Maintains the original state-machine style and adds 'day' (repeated-measures).
"""

import os, sys
import numpy as np
import pandas as pd
from psychopy import visual, core, event, logging # type: ignore
from psychopy.hardware import keyboard # type: ignore
from util_func import *

# --------------------------- EEG (Parallel Port) helper ---------------------------
# Flip-locked rising edges; non-blocking clear to zero a few ms later.
EEG_ENABLED = False
EEG_PORT_ADDRESS = '0x3FD8'
EEG_DEFAULT_PULSE_MS = 10

TRIG = {
    "EXP_START": 10,
    "ITI_ONSET": 11,
    "STIM_ONSET_A": 20,
    "STIM_ONSET_B": 21,
    "RESP_A": 30,
    "RESP_B": 31,
    "FB_COR": 40,
    "FB_INC": 41,
    "EXP_END": 15,
}


class EEGPort:

    def __init__(self,
                 win,
                 address=EEG_PORT_ADDRESS,
                 enabled=EEG_ENABLED,
                 default_ms=EEG_DEFAULT_PULSE_MS):
        self.win = win
        self.enabled = enabled
        self.default_ms = default_ms
        self._port = None
        self._clear_at = None
        if not self.enabled:
            return
        try:
            from psychopy import parallel # type: ignore
            self._port = parallel.ParallelPort(address=address)
        except Exception as e:
            print(
                f"[EEG] Parallel port unavailable ({e}). Running without triggers."
            )
            self.enabled = False
            self._port = None

    def flip_pulse(self, code, width_ms=None, global_clock=None):
        """Schedule a flip-locked pulse: set code on next win.flip, clear after width_ms."""
        if not (self.enabled and self._port):
            return
        width_ms = self.default_ms if width_ms is None else width_ms
        # rising edge exactly on next flip:
        self.win.callOnFlip(self._port.setData, int(code) & 0xFF)
        # schedule a timed clear to 0 after the flip:
        if global_clock is not None:
            # record when to clear (relative to global clock)
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def pulse_now(self, code, width_ms=None, global_clock=None):
        """Immediate pulse (not flip-locked) -- useful for response events."""
        if not (self.enabled and self._port):
            return
        width_ms = self.default_ms if width_ms is None else width_ms
        self._port.setData(int(code) & 0xFF)
        if global_clock is not None:
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def update(self, global_clock=None):
        """Call every frame: clears the port to 0 if a pulse has expired."""
        if not (self.enabled and self._port):
            return
        if self._clear_at is not None and global_clock is not None:
            if global_clock.getTime() >= self._clear_at:
                self._port.setData(0)
                self._clear_at = None

    def close(self):
        try:
            if self._port:
                self._port.setData(0)
        except Exception:
            pass


# ----------------------------------------------------------------------------------

if __name__ == "__main__":

    # --------------------------- Subject / Day handling ---------------------------
    subject = 503
    day = 1

    # NOTE: We want each session to last about 20 min = 20 min * 60 sec / min = 1200 sec.
    #       ITI is 1000 ms + jitter (200-400 ms) = ~1.3 sec
    #       stimulus is response terminated, but assume ~1.5 sec mean rt
    #       feedback is 1000 ms + jitter (200-400 ms) = ~1.3 sec
    #       Total per trial = 1.3 + 1.5 + 1.3 = ~4.1 sec
    #
    #       1200 sec / 4.1 sec = ~292 trials possible in 20 min.
    #       This means we can have up to ~146 stimuli per category per session
    #       Round up to 150 since RT is likely to be faster than 1.5 sec on average.
    n_stimuli_per_category=20

    dir_data = "../data"
    os.makedirs(dir_data, exist_ok=True)
    f_name = f"sub_{subject:03d}_day_{day:02d}_data.csv"
    full_path = os.path.join(dir_data, f_name)

    if os.path.exists(full_path):
        print(f"File {f_name} already exists. Aborting.")
        sys.exit()

    # --------------------------- Stimuli and Categories  ---------------------------
    ds = make_stim_cats(n_stimuli_per_category)
    ds = ds.sample(frac=1).reset_index(drop=True)
    n_trial = ds.shape[0]

    # NOTE: Uncomment to visualize stimulus space scatter
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(6, 6))
    # sns.scatterplot(data=ds, x='x', y='y', hue='cat', ax=ax[0, 0])
    # sns.scatterplot(data=ds, x='xt', y='yt', hue='cat', ax=ax[0, 1])
    # ds['yt_deg'] = ds['yt'] * 180.0 / np.pi
    # sns.scatterplot(data=ds, x='xt', y='yt_deg', hue='cat', ax=ax[0, 2])
    # plt.show()

    # --------------------------- Display / geometry -------------------------------
    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54
    size_cm = 5
    size_px = int(size_cm * px_per_cm)

    win = visual.Window(size=(1920, 1080),
                        fullscr=True,
                        units='pix',
                        color=(0.494, 0.494, 0.494),
                        colorSpace='rgb',
                        winType='pyglet',
                        useRetina=True,
                        waitBlanking=True)

    win.mouseVisible = False
    frame_rate = win.getActualFrameRate()
    print(f"[Info] Frame rate: {frame_rate}")

    center_x, center_y = 0, 0

    # --------------------------- Stim objects ------------------------------------
    fix_h = visual.Line(win,
                        start=(0, -10),
                        end=(0, 10),
                        lineColor='white',
                        lineWidth=8)
    fix_v = visual.Line(win,
                        start=(-10, 0),
                        end=(10, 0),
                        lineColor='white',
                        lineWidth=8)

    init_text = visual.TextStim(win,
                                text="Please press the space bar to begin",
                                color='white',
                                height=32)

    finished_text = visual.TextStim(
        win,
        text="You finished! Thank you for participating!",
        color='white',
        height=32)

    grating = visual.GratingStim(win,
                                 tex='sin',
                                 mask='circle',
                                 interpolate=True,
                                 size=(size_px, size_px),
                                 units='pix',
                                 sf=0.02,
                                 ori=0.0)

    fb_ring = visual.Circle(win,
                            radius=(size_px // 2 + 10),
                            edges=128,
                            fillColor=None,
                            lineColor='white',
                            lineWidth=10,
                            units='pix',
                            pos=(center_x, center_y))

    kb = keyboard.Keyboard()
    default_kb = keyboard.Keyboard()

    global_clock = core.Clock()
    state_clock = core.Clock()
    stim_clock = core.Clock()

    # --------------------------- EEG init ----------------------------------------
    eeg = EEGPort(win)

    # --------------------------- State machine setup ------------------------------
    time_state = 0.0
    state_current = "state_init"
    state_entry = True

    resp = -1
    rt = -1
    trial = -1

    # Record keeping
    trial_data = {
        'subject': [],
        'day': [],
        'trial': [],
        'cat': [],
        'x': [],
        'y': [],
        'xt': [],
        'yt': [],
        'resp': [],
        'rt': [],
        'fb': []
    }

    # --------------------------- Main loop ---------------------------------------
    running = True
    while running:

        if default_kb.getKeys(keyList=['escape'], waitRelease=False):
            running = False
            break

        eeg.update(global_clock)

        # --------------------- STATE: INIT ---------------------
        if state_current == "state_init":
            if state_entry:
                state_clock.reset()
                win.color = (0.494, 0.494, 0.494)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            init_text.draw()

            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True)
            if keys:
                eeg.flip_pulse(TRIG["EXP_START"], global_clock=global_clock)
                state_current = "state_iti"
                state_entry = True

            win.flip()

        # --------------------- STATE: FINISHED ---------------------
        elif state_current == "state_finished":
            if state_entry:
                eeg.flip_pulse(TRIG["EXP_END"], global_clock=global_clock)
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            finished_text.draw()
            win.flip()

        # --------------------- STATE: ITI ---------------------
        elif state_current == "state_iti":
            if state_entry:
                state_clock.reset()
                eeg.flip_pulse(TRIG["ITI_ONSET"], global_clock=global_clock)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            fix_h.draw()
            fix_v.draw()

            if time_state > 1000:
                resp = -1
                rt = -1
                state_clock.reset()
                trial += 1
                if trial >= n_trial:
                    state_current = "state_finished"
                    state_entry = True
                else:
                    sf_cycles_per_cm = ds['xt'].iloc[trial]
                    sf_cycles_per_pix = sf_cycles_per_cm / px_per_cm
                    ori_deg = ds['yt'].iloc[trial] * 180.0 / np.pi
                    cat = ds['cat'].iloc[trial]

                    grating.sf = sf_cycles_per_pix
                    grating.ori = ori_deg
                    grating.pos = (center_x, center_y)

                    kb.clearEvents()
                    state_current = "state_stim"
                    state_entry = True

                    jitter = np.random.randint(200, 401)
                    core.wait(jitter / 1000.0)

            win.flip()

        # --------------------- STATE: STIM ---------------------
        elif state_current == "state_stim":
            if state_entry:
                if cat == "A":
                    trig = TRIG["STIM_ONSET_A"]
                else:
                    trig = TRIG["STIM_ONSET_B"]

                eeg.flip_pulse(trig, global_clock=global_clock)

                state_clock.reset()
                stim_clock.reset()

                win.callOnFlip(kb.clock.reset)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()

            keys = kb.getKeys(keyList=['d', 'k'], waitRelease=False)
            if keys:
                k = keys[-1]
                rt = k.rt * 1000.0
                if k.name == 'd':
                    resp_label = "A"
                    eeg.pulse_now(TRIG["RESP_A"], global_clock=global_clock)
                else:
                    resp_label = "B"
                    eeg.pulse_now(TRIG["RESP_B"], global_clock=global_clock)

                if cat == resp_label:
                    fb = "Correct"
                else:
                    fb = "Incorrect"

                resp = resp_label

                state_clock.reset()
                state_current = "state_feedback"
                state_entry = True

                jitter = np.random.randint(200, 401)
                core.wait(jitter / 1000.0)

            win.flip()

        # --------------------- STATE: FEEDBACK ---------------------
        elif state_current == "state_feedback":
            if state_entry:

                if fb == "Correct":
                    fb_ring.lineColor = 'green'
                    eeg.flip_pulse(TRIG["FB_COR"], global_clock=global_clock)
                else:
                    fb_ring.lineColor = 'red'
                    eeg.flip_pulse(TRIG["FB_INC"], global_clock=global_clock)

                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()
            fb_ring.draw()

            if time_state > 1000:
                trial_data['subject'].append(subject)
                trial_data['day'].append(day)
                trial_data['trial'].append(trial)
                trial_data['cat'].append(ds['cat'].iloc[trial])
                trial_data['x'].append(ds['x'].iloc[trial])
                trial_data['y'].append(ds['y'].iloc[trial])
                trial_data['xt'].append(ds['xt'].iloc[trial])
                trial_data['yt'].append(ds['yt'].iloc[trial])
                trial_data['resp'].append(resp)
                trial_data['rt'].append(rt)
                trial_data['fb'].append(fb)

                pd.DataFrame(trial_data).to_csv(full_path, index=False)

                state_current = "state_iti"
                state_entry = True
                resp = -1
                rt = -1

                jitter = np.random.randint(200, 401)
                core.wait(jitter / 1000.0)

            win.flip()

    # --------------------------- Cleanup ------------------------------------------
    eeg.close()
    win.close()
    core.quit()
