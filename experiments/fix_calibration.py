import numpy as np
import pandas as pd

from src.bn.inference import DelayRiskBN
from src.bn.learn_cpts import load_learned_cpts
from src.data.reason_mapper import factor_ids
from src.eval.calibrate import calibrate_threshold_from_probs
from src.eval.episode_replay import replay_episode
from src.policy.disagreement_voi import DisagreementVoIPolicy
from src.policy.fixed_ranking import FixedRankingPolicy

ep = pd.read_csv("data/processed/episodes.csv")
learned = load_learned_cpts()
factors = factor_ids()
order = sorted(factors, key=lambda f: learned["consensus_priors"][f], reverse=True)
val = ep[ep["split"] == "val"]
ps_d, ps_f, ys = [], [], []
for pid in val["pid"].unique()[:300]:
    g = val[val["pid"] == pid]
    y = int(g["severe_delay"].iloc[0])
    bn_d = DelayRiskBN(learned["disagreement_priors"], learned["p_cause"], learned["leak"])
    bn_f = DelayRiskBN(learned["consensus_priors"], learned["p_cause"], learned["leak"])
    td = replay_episode(DisagreementVoIPolicy(1), bn_d, g, 2, factors, threshold=1.1)
    tf = replay_episode(FixedRankingPolicy(order), bn_f, g, 2, factors, threshold=1.1)
    ps_d.append(td.p_severe)
    ps_f.append(tf.p_severe)
    ys.append(y)
ps_d = np.array(ps_d)
ps_f = np.array(ps_f)
ys = np.array(ys)
print("dis", ps_d.min(), np.median(ps_d), ps_d.max(), "nunique", len(np.unique(np.round(ps_d, 6))))
print("fix", ps_f.min(), np.median(ps_f), ps_f.max(), "nunique", len(np.unique(np.round(ps_f, 6))))
print("cal dis", calibrate_threshold_from_probs(ys, ps_d))
print("cal fix", calibrate_threshold_from_probs(ys, ps_f))
