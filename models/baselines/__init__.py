# Baseline models registry
from .social_lstm    import SocialLSTM
from .social_gan     import SocialGAN
from .trajectronpp   import TrajectronPP
from .social_stgmlp  import SocialSTGMLP
from .d_stgcn        import DSTGCN
from .sgcn           import SGCN
from .gat_baseline   import GATBaseline
from .memonet        import MemoNet
from .social_vae     import SocialVAEBaseline

BASELINE_REGISTRY = {
    "social_lstm":    SocialLSTM,
    "social_gan":     SocialGAN,
    "trajectron++":   TrajectronPP,
    "social_stgmlp":  SocialSTGMLP,   # R1-3: added per reviewer request
    "d_stgcn":        DSTGCN,         # R1-3: added per reviewer request
    "sgcn":           SGCN,           # R1-2: motion primitive comparison
    "gat":            GATBaseline,
    "memonet":        MemoNet,
    "social_vae_fpc": SocialVAEBaseline,
}
