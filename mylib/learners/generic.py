import math
from dataclasses import dataclass
import lightning.pytorch as pl
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class BaseLearnerConfig:
    max_lr: float = 1e-5
    lr_div_factor: float = 10
    batch_size: int = 16
    warmup_pct: float = 0.1
    weight_decay = 1e-2
    betas : tuple = (0.9, 0.999)
    total_steps: int = None
    lr_decay: bool = True


#! fix `get_parameter_group()`
class BaseLearner(pl.LightningModule):
    # config: BaseLearnerConfig

    @staticmethod
    def get_default_config():
        raise NotImplementedError()

    def get_parameter_groups(self):
        """Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, )
        # blacklist_weight_modules = (NoneType,)   #(torch.nn.LayerNorm, torch.nn.Embedding)
        for pn, p in self.named_parameters():

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(pn)
            else:
                decay.add(pn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        return optim_groups

    def configure_optimizers(self):
        optimizer = AdamW(self.get_parameter_groups(), amsgrad=True, lr=self.config.max_lr, betas=self.config.betas)

        def one_cycle_lr(step):
            warmup_pct = self.config.warmup_pct
            total_steps = self.config.total_steps
            warmup_steps = math.floor(warmup_pct * total_steps)
            
            if step < warmup_steps:
                return 1 - 0.5 * (1 - 1 / self.config.lr_div_factor) * (1 + math.cos(step / warmup_steps * math.pi))
            
            if self.config.lr_decay:
                return 1 / self.config.lr_div_factor + 0.5 * (1 - 1 / self.config.lr_div_factor) * (1 + math.cos((step - warmup_steps)/ (total_steps - warmup_steps) * math.pi))

            return 1
        
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=one_cycle_lr
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}