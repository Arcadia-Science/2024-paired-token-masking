amino_acids = sorted(list("ACDEFGHIKLMNPQRSTVWY"))


class ModelName:
    ESM2_8M: str = "facebook/esm2_t6_8M_UR50D"
    ESM2_35M: str = "facebook/esm2_t12_35M_UR50D"
    ESM2_150M: str = "facebook/esm2_t30_150M_UR50D"
    ESM2_650M: str = "facebook/esm2_t33_650M_UR50D"
    ESM2_3B: str = "facebook/esm2_t36_3B_UR50D"
    ESM2_15B: str = "facebook/esm2_t48_15B_UR50D"

    @classmethod
    def all(cls) -> list[str]:
        return [getattr(cls, name) for name in cls.__annotations__]
