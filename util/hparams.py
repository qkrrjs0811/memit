import json
from dataclasses import dataclass, fields


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)


    def update_from_dict(self, hparams_mod: dict):
        for field in fields(self):
            field_name = field.name

            if field_name in hparams_mod.keys():
                setattr(self, field_name, hparams_mod[field_name])

