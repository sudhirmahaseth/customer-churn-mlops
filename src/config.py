from dataclasses import dataclass

@dataclass
class TrainingConfig:

    test_size: float = 0.2

    random_state: int = 42

    model_path: str = "models/model.pkl"

    target_column: str = "Churn"