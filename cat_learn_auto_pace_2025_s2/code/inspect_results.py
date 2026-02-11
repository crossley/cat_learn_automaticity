"""
Main behavioral analysis script for the longitudinal category-learning automaticity study
(cat_learn_auto_pace_2025_s2).

Overview
--------
Participants practice a procedural categorization task longitudinally:
- At-home training across many days
- Interleaved special at-home sessions:
    * Dual-task day (day 22)
    * Button-switch days (days 23–24)
- Lab sessions approximately every ~5 days (behavior + EEG recorded; EEG analysis lives elsewhere)

This script:
1) Loads at-home CSVs from ../data/*/sub_*_day_*_data.csv with manual corrections/exclusions.
2) Loads lab behavioral CSVs from ../data_lab_behave/*.csv with a few filename/day fixes.
3) Builds unified long-format dataframe across session types:
      - Training at home
      - Dual-Task at home
      - Button-Switch at home
      - Training in the Lab
4) Applies inclusion criteria:
      - keep only subjects with data present in ALL session types
      - Stroop (ns_*) accuracy >= 0.80 (computed where ns_* columns exist)
      - RT filter for plotting (rt <= 3000 ms)
5) Fits decision-bound models (DBM) per subject × day (block_size=25 trials) if
   ../dbm_fits/dbm_results.csv does not already exist; selects best model by BIC and collapses to:
      - procedural (GLC)
      - rule-based (unidimensional)
6) Produces figures in ../figures/:
      - DBM BIC by day and best_model_class
      - Accuracy by day across session types
      - RT by day across session types
      - Dual-task: last training vs dual-task day (accuracy + RT)
      - Button-switch: last training vs button-switch days (accuracy + RT)
      - EEG predictions placeholder figure (synthetic)

Data organization assumptions
-----------------------------
At-home data are organized by participant folder, e.g.:
  ../data/subj_<hash>/sub_<ID>_day_<NN>_data.csv

Lab behavioral files live in:
  ../data_lab_behave/sub_<ID>_day_<NNN>_data.csv

The script contains a manual "day exclusion list" and several subject-specific fixes
(mislabeled subject IDs, mislabeled day numbers, extra days). These are documented inline.
"""

from imports import *
from util_func_dbm import *

if __name__ == '__main__':

    # NOTE: Init figure style
    sns.set_palette("rocket", n_colors=4)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    dir_data = "../data"
    dir_data_lab_beh = "../data_lab_behave"

    df_train_rec = []  # training days
    df_bs_rec = []  # button switch days
    df_dt_rec = []  # dual task days
    df_lab_rec = []  # lab days

    for fd in os.listdir(dir_data):
        dir_data_fd = os.path.join(dir_data, fd)
        if os.path.isdir(dir_data_fd):
            for fs in os.listdir(dir_data_fd):
                f_full_path = os.path.join(dir_data_fd, fs)
                if os.path.isfile(f_full_path):

                    # 1. sub 002 has an at home day labelled day 23
                    #    that was also the 17th at home day
                    #    (i.e., an extra day)
                    # 2. sub 008 did four extra days
                    #    they didn't understand EEG days counted
                    # 3. sub 015 missed 2 at home days, therefore,
                    #    excluding days 22, 23, 24
                    # 4. sub 019 did 2 extra days. Day 13 froze their
                    #    computer during the task (84 trials completed).
                    #    They continued with the next day instead
                    #    of re-doing day 13. So, D13 is unusable and
                    #    will be excluded here. Will also exclude day 18,
                    #    because it is an extra at home day.
                    #
                    # NOTE: Day Exclusion List
                    if fs not in [
                            'sub_002_day_23_data.csv',  # extra at home day
                            'sub_008_day_18_data.csv',  # extra at home day
                            'sub_008_day_19_data.csv',  # extra at home day
                            'sub_008_day_20_data.csv',  # extra at home day
                            'sub_008_day_21_data.csv',  # extra at home day
                            'sub_008_day_22_data.csv',  # exclude DT
                            'sub_008_day_23_data.csv',  # exclude BS
                            'sub_008_day_24_data.csv',  # exclue BS
                            'sub_015_day_22_data.csv',  # exclude DT
                            'sub_015_day_23_data.csv',  # exclude BS
                            'sub_015_day_24_data.csv',  # exclude BS
                            'sub_019_day_13_data.csv',  # computer froze
                            'sub_019_day_18_data.csv',  # exclude DT
                            'sub_019_day_22_data.csv',  # exclude BS
                            'sub_019_day_23_data.csv',  # exclude BS
                            'sub_019_day_24_data.csv'
                    ]:  # extra at home day

                        df = pd.read_csv(f_full_path)
                        df['f_name'] = fs

                        # sub_003 somehow replicated day 18
                        # fix that here
                        if fs == 'sub_003_day_19_data.csv':
                            df['day'] = 19
                        elif fs == 'sub_003_day_20_data.csv':
                            df['day'] = 20

                        # NOTE: sub_006 mislabeled days 22, 23, and 24 as sub_001
                        # manually change file name to: sub_006_day_22_data.csv,
                        # sub_006_day_23_data.csv, sub_006_day_24_data.csv
                        # fix that here
                        if fs == 'sub_006_day_22_data.csv':
                            df['subject'] = 6
                        elif fs == 'sub_006_day_23_data.csv':
                            df['subject'] = 6
                        elif fs == 'sub_006_day_24_data.csv':
                            df['subject'] = 6

                        # fix sub_015 mislabeling in raw data
                        # also fix extra 1 in day 7 (trial 60 or 61) manually
                        if fs == 'sub_015_day_01_data.csv':
                            df['subject'] = 15

                        # sub_016 mislabeled day 22 as sub_001 and had already
                        # changed file name to: sub_016_day_22_data.csv
                        # changing from subject 1 to subject 16
                        # fix that here
                        if fs == 'sub_016_day_22_data.csv':
                            df['subject'] = 16

                        # NOTE: sub_017 day 17 mislabeled to sub_007_day_17_data.csv
                        # manually change file name to: sub_017_day_17_data.csv
                        # fix subject column here
                        if fs == 'sub_017_day_17_data.csv':
                            df['subject'] = 17

                        # NOTE: sub_019
                        # mislabeled day 1 as sub_001
                        # no need to manually change file name
                        # fix that here
                        if fs == 'sub_019_day_01_data.csv':
                            df['subject'] = 19

                        # from trial 276 on day 24, 'day' changes to day 25
                        # (until the end of the expt)
                        # also, in this file there are 4 lines of ,,,,,,,,,,,
                        # after the end of the experiment
                        # manually removed those lines
                        # fix that here
                        if fs == 'sub_019_day_24_data.csv':
                            df['day'] = 24

                        day = df['day'].unique()

                        # training days
                        if ~np.isin(day, [22, 23, 24]):
                            df_train_rec.append(df)

                        # dual-task days
                        if day == 22:
                            df_dt_rec.append(df)

                        # button-switch days
                        if day in [23, 24]:
                            df_bs_rec.append(df)

    for fd in os.listdir(dir_data_lab_beh):
        f_df = os.path.join(dir_data_lab_beh, fd)
        if os.path.isfile(f_df) and fd != '.DS_Store':
            df = pd.read_csv(f_df)

            # mislabelled sub_003_day_401_data.csv
            # manually changed file name to: sub_003_day_403_data.csv
            # fix here
            if fd == 'sub_003_day_401_data.csv':
                df['day'] = 403

            # mislabelled sub_015_day_15_data.csv
            # manually changed file name to: sub_015_day_215_data.csv
            # fix here
            if fd == 'sub_015_day_15_data.csv':
                df['day'] = 215

            df_lab_rec.append(df)

    block_size = 25

    d = pd.concat(df_train_rec, ignore_index=True)
    d.sort_values(by=['subject', 'day', 'trial'], inplace=True)
    d['acc'] = (d['cat'] == d['resp']).astype(int)
    d['day'] = d.groupby('subject')['day'].rank(method='dense').astype(int)
    d['trial'] = d.groupby(['subject']).cumcount()
    d['n_trials'] = d.groupby(['subject', 'day'])['trial'].transform('count')
    d['block'] = d.groupby(['subject', 'day'
                            ])['trial'].transform(lambda x: x // block_size)
    d['session_type'] = 'Training at home'

    d_dt = pd.concat(df_dt_rec, ignore_index=True)
    d_dt.sort_values(by=['subject', 'day', 'trial'], inplace=True)
    d_dt['acc'] = (d_dt['cat'] == d_dt['resp']).astype(int)
    d_dt['day'] = d_dt.groupby('subject')['day'].rank(
        method='dense').astype(int)
    d_dt['day'] = d_dt['day'].map({1: 22})
    d_dt['trial'] = d_dt.groupby(['subject']).cumcount()
    d_dt['n_trials'] = d_dt.groupby(['subject',
                                     'day'])['trial'].transform('count')
    d_dt['session_type'] = 'Dual-Task at home'

    d_bs = pd.concat(df_bs_rec, ignore_index=True)
    d_bs.sort_values(by=['subject', 'day', 'trial'], inplace=True)
    d_bs['acc'] = (d_bs['cat'] == d_bs['resp']).astype(int)
    d_bs['day'] = d_bs.groupby('subject')['day'].rank(
        method='dense').astype(int)
    d_bs['day'] = d_bs['day'].map({1: 23, 2: 24})
    d_bs['trial'] = d_bs.groupby(['subject']).cumcount()
    d_bs['n_trials'] = d_bs.groupby(['subject',
                                     'day'])['trial'].transform('count')
    d_bs['session_type'] = 'Button-Switch at home'

    d_lab = pd.concat(df_lab_rec, ignore_index=True)
    d_lab['acc'] = (d_lab['cat'] == d_lab['resp']).astype(int)
    d_lab['day'] = d_lab.groupby('subject')['day'].rank(
        method='dense').astype(int)
    d_lab['day'] = d_lab['day'].map({1: 0.5, 2: 4.5, 3: 8.5, 4: 12.5, 5: 21})
    d_lab['trial'] = d_lab.groupby(['subject']).cumcount()
    d_lab['n_trials'] = d_lab.groupby(['subject',
                                       'day'])['trial'].transform('count')
    d_lab['block'] = d_lab.groupby(
        ['subject', 'day'])['trial'].transform(lambda x: x // block_size)
    d_lab['session_type'] = 'Training in the Lab'

    # NOTE: create a numpy array of the intersection of subjects across all dataframes
    all_subs = np.unique(
        np.concatenate([
            d.subject.unique(),
            d_dt.subject.unique(),
            d_bs.subject.unique(),
            d_lab.subject.unique()
        ]))

    subs_to_keep = np.intersect1d(all_subs, d.subject.unique())
    subs_to_keep = np.intersect1d(subs_to_keep, d_dt.subject.unique())
    subs_to_keep = np.intersect1d(subs_to_keep, d_bs.subject.unique())
    subs_to_keep = np.intersect1d(subs_to_keep, d_lab.subject.unique())

    # merge all dataframes inserting np.nan into columns that don't exist in a particular dataframe
    d_all = pd.concat([d, d_dt, d_bs, d_lab], ignore_index=True, sort=False)
    d_all['day'] = d_all.groupby('subject')['day'].rank(
        method='dense').astype(int)

    # exclude subjects not in all three dataframes
    d_all = d_all[d_all['subject'].isin(subs_to_keep)].reset_index(drop=True)

    # NOTE: compute Stroop accuracy and exlcude subjects with accuracy < 80%
    d_all['acc_stroop'] = np.nan
    d_all.loc[d_all['ns_correct_side'].notna(), 'acc_stroop'] = (
        d_all['ns_correct_side'] == d_all['ns_resp']).astype(int)
    d_all['acc_stroop_mean'] = d_all.groupby(
        'subject')['acc_stroop'].transform(lambda x: np.nanmean(x))
    d_all = d_all[d_all['acc_stroop_mean'] >= 0.8].reset_index(drop=True)

    # NOTE: primary exclusion criteria for remaining subjects will be deciion bound fits
    #       Fit DBM here
    models = [
        nll_unix,
        nll_unix,
        nll_uniy,
        nll_uniy,
        nll_glc,
        nll_glc,
    ]
    side = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
    k = [2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    n = block_size
    model_names = [
        "nll_unix_0",
        "nll_unix_1",
        "nll_uniy_0",
        "nll_uniy_1",
        "nll_glc_0",
        "nll_glc_1",
    ]

    if not os.path.exists("../dbm_fits/dbm_results.csv"):
        dbm = (d.groupby(["subject", "day"]).apply(fit_dbm, models, side, k, n,
                                                   model_names).reset_index())
        dbm.to_csv("../dbm_fits/dbm_results.csv")
    else:
        dbm = pd.read_csv("../dbm_fits/dbm_results.csv")
        dbm = dbm[["subject", "day", "model", "bic", "p"]]

    def assign_best_model(x):
        model = x["model"].to_numpy()
        bic = x["bic"].to_numpy()
        best_model = np.unique(model[bic == bic.min()])[0]
        x["best_model"] = best_model
        return x

    dbm = dbm.groupby(["subject",
                       "day"]).apply(assign_best_model,
                                     include_groups=False).reset_index()
    dbm = dbm[dbm["model"] == dbm["best_model"]]
    dbm = dbm[["subject", "day", "bic", "best_model"]]
    dbm = dbm.drop_duplicates().reset_index(drop=True)
    dbm["best_model_class"] = dbm["best_model"].str.split("_").str[1]
    dbm.loc[dbm["best_model_class"] != "glc",
            "best_model_class"] = "rule-based"
    dbm.loc[dbm["best_model_class"] == "glc",
            "best_model_class"] = "procedural"
    dbm["best_model_class"] = dbm["best_model_class"].astype("category")
    dbm = dbm.reset_index(drop=True)

    # print proportion of best model classes across all subjects and days
    dbm.groupby('day')['best_model_class'].value_counts(normalize=True)

    # NOTE: plot bic across days for each model class
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 7))
    sns.pointplot(data=dbm,
                  x='day',
                  y='bic',
                  hue='best_model_class',
                  errorbar=('se'),
                  ax=ax[0, 0])
    ax[0, 0].set_xlabel('Day')
    ax[0, 0].set_ylabel('BIC')
    plt.tight_layout()
    plt.savefig('../figures/dbm_bic_performance.png', dpi=300)

    # NOTE: aggregate data for upcoming figures
    d_all = d_all[d_all['rt'] <= 3000]
    dd_all = d_all.groupby(['subject', 'day', 'session_type']).agg({
        'acc': 'mean',
        'rt': 'mean'
    }).reset_index()

    # NOTE: Figure --- all session types
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
    sns.pointplot(data=dd_all,
                  x='day',
                  y='acc',
                  hue='session_type',
                  errorbar=('se'),
                  ax=ax[0, 0])
    [
        x.set_xticks(np.arange(0, dd_all['day'].max() + 2, 1))
        for x in ax.flatten()
    ]
    ax[0, 0].set_xlabel('Day', fontsize=16)
    ax[0, 0].set_ylabel('Proportion correct', fontsize=16)
    ax[0, 0].legend(title='')
    plt.savefig('../figures/training_performance_days.png', dpi=300)
    plt.close()

    # NOTE: Figure --- all session types RT
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
    sns.pointplot(data=dd_all,
                  x='day',
                  y='rt',
                  hue='session_type',
                  errorbar=('se'),
                  ax=ax[0, 0])
    [
        x.set_xticks(np.arange(0, dd_all['day'].max() + 2, 1))
        for x in ax.flatten()
    ]
    ax[0, 0].set_xlabel('Day', fontsize=16)
    ax[0, 0].set_ylabel('Reaction Time', fontsize=16)
    ax[0, 0].legend(title='')
    plt.savefig('../figures/training_performance_days_rt.png', dpi=300)
    plt.close()

    # NOTE: dual-task figures
    # prepare a data frame comparing last day of training to dual-task day
    d_dtf = dd_all[dd_all['day'].isin([20, 22])].copy()

    # change the day column to categorical for plotting with names "Last Training Day" and "Dual-Task Day"
    d_dtf['day'] = d_dtf['day'].map({
        20: 'Last Training Day',
        22: 'Dual-Task Day'
    })

    # plot point range plot comparing the last day of training to dual-task day
    fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(5, 8))
    sns.pointplot(data=d_dtf, x='day', y='acc', errorbar=('se'), ax=ax[0, 0])
    sns.pointplot(data=d_dtf, x='day', y='rt', errorbar=('se'), ax=ax[1, 0])
    ax[0, 0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    ax[0, 0].set_xlabel('')
    ax[0, 0].set_ylabel('Accuracy (proportion correct)')
    ax[1, 0].set_xlabel('')
    ax[1, 0].set_ylabel('Reaction Time (ms)')
    plt.tight_layout()
    plt.savefig('../figures/dual_task_performance.png', dpi=300)
    plt.close()

    # NOTE: dual-task stats
    res = pg.ttest(x=d_dtf[d_dtf['day'] == 'Last Training Day']['acc'],
                   y=d_dtf[d_dtf['day'] == 'Dual-Task Day']['acc'],
                   alternative='greater',
                   paired=True)

    # NOTE: button-switch figures

    # prepare a data frame comparing last day of training to button-switch days
    d_bsf = dd_all[dd_all['day'].isin([20, 23, 24])].copy()

    # change the day column to categorical for plotting with names "Last Training Day", "Button-Switch Day 1", "Button-Switch Day 2"
    d_bsf['day'] = d_bsf['day'].map({
        20: 'Last Training Day',
        23: 'Button-Switch Day 1',
        24: 'Button-Switch Day 2'
    })

    # plot point range plot comparing the last day of training to button-switch days
    fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(7, 8))
    sns.pointplot(data=d_bsf, x='day', y='acc', errorbar=('se'), ax=ax[0, 0])
    sns.pointplot(data=d_bsf, x='day', y='rt', errorbar=('se'), ax=ax[1, 0])
    ax[0, 0].set_xlabel('')
    ax[0, 0].set_ylabel('Accuracy (proportion correct)')
    ax[1, 0].set_xlabel('')
    ax[1, 0].set_ylabel('Reaction Time (ms)')
    plt.tight_layout()
    plt.savefig('../figures/button_switch_performance.png', dpi=300)
    plt.close()

    # NOTE: button-switch stats
    res_bs1 = pg.ttest(x=d_bsf[d_bsf['day'] == 'Last Training Day']['acc'],
                       y=d_bsf[d_bsf['day'] == 'Button-Switch Day 1']['acc'],
                       alternative='greater',
                       paired=True)

    res_bs2 = pg.ttest(x=d_bsf[d_bsf['day'] == 'Last Training Day']['acc'],
                       y=d_bsf[d_bsf['day'] == 'Button-Switch Day 2']['acc'],
                       alternative='greater',
                       paired=True)

    # NOTE: Make EEG predictions figure
    # draw 5 sets of two gaussians one centered at 500 ms and another centered at 1000 ms
    # let the amplitude of these gaussian increase across the 5 sets, but at different rates for
    # each centre. 
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
    x = np.linspace(0, 1500, 1000)
    for i in range(5):
        y1 = (2*i + 1) * np.exp(-0.5 * ((x - 500) / 100)**2)
        y2 = (i + 2) * np.exp(-0.5 * ((x - 1000) / 100)**2)
        ax[0, 0].plot(x, y1 + y2, label=f'Set {i+1}')
    ax[0, 0].set_xlabel('Time within trial (ms)', fontsize=16)
    ax[0, 0].set_ylabel('Functional Connectivity (a.u.)', fontsize=16)
    ax[0, 0].legend().remove()
    ax[0, 0].legend([f'Day {i+1}' for i in range(5)], title='')
    plt.savefig('../figures/eeg_predictions.png', dpi=300)
    plt.close()
