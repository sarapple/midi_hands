import torch
from midi_hands.utils import Utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Mobile():
    @staticmethod
    def save_model_to_mobile_file(model=None, model_name=""):
        model = model.eval()
        # writer = SummaryWriter()
        # writer.add_graph(model, verbose=True, input_to_model=torch.rand((100, 16, 4), dtype=torch.float))
        # writer.close()
        example = torch.rand(100, 16, 4)
        # for param in model.parameters():
            # param.requires_grad = False
        with torch.no_grad():
            traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(Utils.build_dir_path(f"mobile/model_{model_name}.pt"))
