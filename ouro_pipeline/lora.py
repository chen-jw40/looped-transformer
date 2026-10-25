"""Sequential LoRA: one fixed-rank q/v adapter trained across every task."""
import torch

from ouro_pipeline.olora import ContinualOLoRA

PAPER_URL = "https://arxiv.org/abs/2106.09685"
REFERENCE_URL = "https://github.com/microsoft/LoRA"


class ContinualLoRA(ContinualOLoRA):
    def __init__(self, model, rank=8, alpha=32., dropout=.1):
        super().__init__(model, rank, alpha, dropout, orthogonality_weight=0., norm_weight=0.)

    def begin_task(self, stage):
        if stage != self.stage + int(self.consolidated):
            raise ValueError("Sequential LoRA tasks must be visited in order")
        self.stage, self.consolidated = stage, False

    def consolidate(self):
        if self.consolidated:
            raise ValueError("LoRA stage already consolidated")
        # Keep exactly the same trainable A/B; do not append or reinitialize.
        self.consolidated = True

    def diagnostics(self):
        return {"variant": "one fixed-rank q/v adapter, sequential tasks",
                "physical_target_modules": len(self.modules),
                "rank_per_module": {name: module.rank for name, module in self.modules.items()},
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "historical_adapters": 0, "regularization": False}

    def adapter_state(self):
        if not self.consolidated:
            raise ValueError("Complete the current task before exporting adapter state")
        return {"format": "sequential_lora_v1", "stage": self.stage,
                "modules": {name: {"A": module.A.detach().cpu().clone(),
                                    "B": module.B.detach().cpu().clone(), "scaling": module.scaling}
                            for name, module in self.modules.items()}}

    @torch.no_grad()
    def load_adapter_state(self, state):
        if state.get("format") != "sequential_lora_v1" or state["modules"].keys() != self.modules.keys():
            raise ValueError("Incompatible sequential LoRA state")
        for name, module in self.modules.items():
            saved = state["modules"][name]
            if saved["scaling"] != module.scaling or saved["A"].shape != module.A.shape or saved["B"].shape != module.B.shape:
                raise ValueError("LoRA rank or scaling changed")
            module.A.copy_(saved["A"])
            module.B.copy_(saved["B"])
        self.stage, self.consolidated = state["stage"], True
