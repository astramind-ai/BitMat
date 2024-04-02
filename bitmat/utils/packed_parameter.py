import torch


class PackedParameter:
    def __init__(self, data: torch.Tensor):
        # Assicurati che i dati siano salvati come int8.
        self.data = data.to(torch.int8)
        self.device = self.data.device
    def __repr__(self):
        return f'PackedParameter containing:\n{str(self.unpack())}'

    def to(self, *args, **kwargs):
        self.data = self.data.to(*args, **kwargs)
        return self
