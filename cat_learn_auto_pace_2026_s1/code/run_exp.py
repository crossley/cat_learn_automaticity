# -*- coding: utf-8 -*-
"""
Single-phase categorization task (PsychoPy-only) with veridical feedback and EEG triggers.
Maintains the original state-machine style and adds 'day' (repeated-measures).
"""

import os
import sys
import uuid
from datetime import datetime
import json
import numpy as np
import pandas as pd
from psychopy import visual, core, event, logging  # type: ignore
from psychopy.hardware import keyboard  # type: ignore
from util_func import *


# --------------------------- EEG (Parallel Port) helper ---------------------------
# Flip-locked rising edges; non-blocking clear to zero a few ms later.
EEG_ENABLED = False
EEG_PORT_ADDRESS = '0x3FD8'
EEG_DEFAULT_PULSE_MS = 10

TRIG = {

    # -------------------- Experiment structure --------------------
    "EXP_START": 10,
    "ITI_ONSET": 11,
    "EXP_END": 15,

    # -------------------- Stimulus onset --------------------
    # Training trials
    "STIM_ONSET_A_TRAIN": 20,
    "STIM_ONSET_B_TRAIN": 21,

    # Probe trials
    "STIM_ONSET_A_PROBE": 22,
    "STIM_ONSET_B_PROBE": 23,

    # -------------------- Responses --------------------
    # Training trials
    "RESP_A_TRAIN": 30,
    "RESP_B_TRAIN": 31,

    # Probe trials
    "RESP_A_PROBE": 32,
    "RESP_B_PROBE": 33,

    # -------------------- Feedback --------------------
    # Training trials
    "FB_COR_TRAIN": 40,
    "FB_INC_TRAIN": 41,

    # Probe trials
    "FB_COR_PROBE": 42,
    "FB_INC_PROBE": 43,
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
            from psychopy import parallel  # type: ignore
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

    # --------------------------- Experiment parameters ---------------------------
    n_train = 550
    n_test = 100
    n_total = n_train + n_test

    # --------------------------- Subject / Day handling ---------------------------
    dir_data = "../data"
    os.makedirs(dir_data, exist_ok=True)

    info_path = os.path.join(dir_data, "participant_info.json")

    if os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            subject = info["subject"]
            condition = int(info["condition"])
        except Exception:
            print("Could not read participant_info.json. Aborting.")
            sys.exit()
    else:
        subject = uuid.uuid4().hex[:12]
        condition = int(np.random.choice([90, 180]))
        info = {"subject": subject, "condition": condition}
        try:
            with open(info_path, "x") as f:
                json.dump(info, f, indent=4)
        except FileExistsError:
            print("participant_info.json already exists. Aborting.")
            sys.exit()

    today_key = datetime.now().strftime("%Y%m%d")

    prefix = f"sub_{subject}_date_{today_key}_"
    suffix = "_data.csv"

    existing = [
        fn for fn in os.listdir(dir_data)
        if fn.startswith(prefix) and fn.endswith(suffix)
    ]

    if len(existing) > 1:
        print("Multiple data files found for today:")
        for f in existing:
            print("  ", f)
        print("Please contact the experimenter.")
        sys.exit()

    if len(existing) == 1:
        f_name = existing[0]
        full_path = os.path.join(dir_data, f_name)
        try:
            n_done = pd.read_csv(full_path).shape[0]
        except Exception:
            print(f"Could not read existing file: {f_name}. Aborting.")
            sys.exit()
    else:
        f_name = f"sub_{subject}_date_{today_key}_data.csv"
        full_path = os.path.join(dir_data, f_name)
        n_done = 0

    if n_done >= n_total:
        print(f"Today's session is already complete ({n_done} trials). Aborting.")
        sys.exit()

    print(
        f"Subject: {subject} | Condition: {condition} | Date: {today_key} | Resuming at trial: {n_done}"
    )

    trial = n_done - 1

    # --------------------------- Stimuli and Categories  ---------------------------
    n_stimuli_per_category = n_total // 2
    ds, ds_90, ds_180 = make_stim_cats(n_stimuli_per_category)

    ds_train = ds.copy()
    ds_train = ds_train.sample(frac=1).reset_index(drop=True)
    ds_train = ds_train.iloc[:n_train, :]
    ds_train["phase"] = "train"

    if condition == 90:
        ds_test = ds_90.copy()
    elif condition == 180:
        ds_test = ds_180.copy()

    ds_test = ds_test.sample(frac=1).reset_index(drop=True)
    ds_test = ds_test.iloc[:n_test, :]
    ds_test["phase"] = "test"

    ds = pd.concat([ds_train, ds_test]).reset_index(drop=True)

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
                if trial >= n_total:
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
                if ds['phase'].iloc[trial] == 'train':
                    if cat == "A":
                        trig = TRIG["STIM_ONSET_A_TRAIN"]
                    else:
                        trig = TRIG["STIM_ONSET_B_TRAIN"]
                elif ds['phase'].iloc[trial] == 'test':
                    if cat == "A":
                        trig = TRIG["STIM_ONSET_A_PROBE"]
                    else:
                        trig = TRIG["STIM_ONSET_B_PROBE"]

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
                if ds['phase'].iloc[trial] == 'train':
                    if k.name == 'd':
                        resp_label = "A"
                        eeg.pulse_now(TRIG["RESP_A_TRAIN"],
                                      global_clock=global_clock)
                    else:
                        resp_label = "B"
                        eeg.pulse_now(TRIG["RESP_B_TRAIN"],
                                      global_clock=global_clock)
                elif ds['phase'].iloc[trial] == 'test':
                    if k.name == 'd':
                        resp_label = "A"
                        eeg.pulse_now(TRIG["RESP_A_PROBE"],
                                      global_clock=global_clock)
                    else:
                        resp_label = "B"
                        eeg.pulse_now(TRIG["RESP_B_PROBE"],
                                      global_clock=global_clock)

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
                if ds['phase'].iloc[trial] == 'train':
                    if fb == "Correct":
                        fb_ring.lineColor = 'green'
                        eeg.flip_pulse(TRIG["FB_COR_TRAIN"],
                                       global_clock=global_clock)
                    else:
                        fb_ring.lineColor = 'red'
                        eeg.flip_pulse(TRIG["FB_INC_TRAIN"],
                                       global_clock=global_clock)
                elif ds['phase'].iloc[trial] == 'test':
                    if fb == "Correct":
                        fb_ring.lineColor = 'green'
                        eeg.flip_pulse(TRIG["FB_COR_PROBE"],
                                       global_clock=global_clock)
                    else:
                        fb_ring.lineColor = 'red'
                        eeg.flip_pulse(TRIG["FB_INC_PROBE"],
                                       global_clock=global_clock)

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
